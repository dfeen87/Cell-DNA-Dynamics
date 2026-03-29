# A Triadic Stability Framework for Detecting Regime Transitions in Time-Resolved Biological Systems

**Medinformatics** | DOI: 10.47852/bonviewMEDINXXXXXXXX

**Marcel Krüger¹\*** · **Don Michael Feeney Jr.²**

¹ Independent Researcher, Germany
² Independent Researcher, USA

\* Corresponding author: marcelkrueger092@gmail.com | dfeen87@gmail.com
ORCID (M.K.): 0009-0002-5709-9729 | ORCID (D.M.F.): 0009-0003-1350-4160

---

## Abstract

Biological systems exhibit a wide range of dynamical transitions associated with instability, repair, regulation, and regeneration. Representative examples include neural collapse, cardiac rhythm transitions, respiratory instability, cellular stress responses, chromatin reorganization, and DNA-repair dynamics. These processes are inherently time-dependent, multivariate, and involve coupled changes in metabolic activity, informational structure, and dynamical coherence.

In this work, we introduce a **triadic stability framework** in which biological dynamics are represented in a reduced state space defined by **Energy, Information, and Coherence (ICE)**. Within this representation, a structured spiral-time coordinate is used as an embedding variable for time-resolved analysis. The resulting instability magnitude **ΔΦ(t)** quantifies deviations of the system from a reference state and serves as an operational indicator for regime-transition detection.

The framework is explicitly conservative in scope: it does not posit new biological forces, does not replace established theories such as molecular biology or statistical physics, and does not model static DNA sequence structure directly. Instead, it targets time-dependent observables, including chromatin conformational dynamics, transcriptional bursting, DNA repair trajectories, replication-stress markers, mitochondrial fluctuations, calcium synchronization, and respiratory oscillations.

We formulate the triadic representation, define the embedding structure, derive the instability magnitude, and propose a staged validation protocol including synthetic simulations, surrogate-data tests, and application to experimental time series. The objective is not to advance therapeutic claims, but to establish a falsifiable and reproducible framework for analyzing regime transitions in complex biological systems.

**Keywords:** triadic stability framework, biological dynamics, cellular stability, DNA-associated dynamics, regime transitions, energy–information–coherence, complex systems modeling, spiral-time embedding

---

## Figure 1 — Conceptual Embedding (Schematic)

```
SPIRAL-TIME EMBEDDING Ψ(t)              ICE STATE SPACE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━           ━━━━━━━━━━━━━━━━━━━━━━━━
                                                    C
 Cytoplasmic membrane                              /|\
 ┌──────────────────────────┐                    / | \
 │  Angular coord φ:        │                   /  |  \
 │  • Chromatin fluctuations│                  /   |   \
 │  • Supercoiling state    │                 / trajectory\
 │                          │                /    ~~~~    \
 │  [DNA strand spiral      │               /   ~Φc─────~  \
 │   schematic]             │              /  ~           ~  \
 │   ≋≋≋≋≋≋≋≋≋≋≋           │             / ~  X*→→X(t)   ~  \
 │                          │            /~  (excursion ΔΦ)~   \
 │  Axial coord t:          │           E ─────────────────────I
 │  • Polymerase traversal  │
 │  • Replication fork      │       ΔΦ(t) = √(Eₙ² + Iₙ² + Cₙ²)
 │                          │
 │  Radial coord χ:         │       ΔΦ(t)↑
 │  • Locus mobility        │       ░░░░░░░░░░░▓▓▓▓▓▓████████
 │  • Loop extrusion        │       stable   trans-  unstable
 │  • Repair focus form.    │                ition
 └──────────────────────────┘                        → t
```

*Left: Schematic mapping of DNA-associated dynamical processes into spiral-time embedding Ψ(t), interpreted strictly as a geometric coordinate system. Axial coordinate t = temporal progression; angular coordinate φ = phase-linked oscillatory structure; radial coordinate χ = deviation/mobility dynamics. Right top: Example trajectory in reduced ICE state space illustrating excursions away from reference stability region toward critical threshold Φc. Right bottom: Illustrative instability measure ΔΦ(t). All panels are conceptual embeddings and do not imply new biophysical structures, forces, or ontological claims about biological time.*

---

## 1 Introduction

Biological systems are inherently dynamic. Their behavior arises from coupled processes spanning multiple temporal and spatial scales, including oscillation, synchronization, collapse, adaptation, repair, and recovery. Many of these processes are not adequately captured by static descriptors or single-variable biomarkers, but instead manifest as **transitions between relatively stable and unstable dynamical regimes**.

Such transitions occur across multiple domains, including pathological shifts in neural activity, rhythm instabilities in physiological systems, metabolic stress responses in cells, chromatin remodeling under perturbation, and DNA-repair dynamics following damage. Recent generative whole-brain modeling work further supports the view that subject-specific functional alterations can be studied as dynamical deviations in coupled multivariate systems [11]. Although these systems differ in mechanism and scale, they share a common structural feature: their evolution is governed by time-dependent interactions among multiple coupled observables.

A central challenge is therefore to construct a reduced yet dynamically meaningful representation in which such regime transitions can be analyzed in a unified and system-agnostic manner. In the present work, we introduce such a representation based on a **triadic state space defined by Energy, Information, and Coherence (ICE)**. The central premise is not that these quantities exhaust biological complexity, but that they provide a minimal and operational scaffold for describing a broad class of stability transitions.

To analyze the temporal organization of these dynamics, we introduce a **spiral-time coordinate** as a structured embedding variable. This formalism is deliberately conservative and should not be interpreted as a claim about the fundamental nature of biological time. Rather, spiral-time serves as a coordinate system for organizing time-resolved observables and for constructing a stability measure sensitive to regime transitions.

The contribution of this work is therefore **methodological**. We formulate a triadic stability framework that provides:
1. a reduced representation of multivariate biological dynamics,
2. an instability magnitude for quantifying deviations from a reference regime, and
3. a structured validation protocol based on synthetic simulations, surrogate controls, and experimental data.

The goal is not to introduce a mechanistic or therapeutic theory, but to establish a falsifiable and reproducible modeling framework for analyzing regime transitions in cellular and DNA-associated systems.

### Relation to Prior Operator-Based Frameworks

The present work builds upon a sequence of operator-based approaches for time-resolved physiological signal analysis.

- The **neural component** originates from information-theoretic modeling of neural coherence using triadic spiral-time embeddings [1].
- This framework was subsequently extended to a **coupled heart–brain architecture** integrating EEG and ECG/HRV dynamics within a unified instability measure [2].
- The current study represents a further methodological extension, focusing on **cellular and DNA-associated dynamics** while preserving the same reduced ICE representation and operator structure.

These works should be interpreted as a progressive development: (i) neural operator formulation → (ii) multimodal physiological extension → (iii) cellular and DNA-associated regime-transition modeling.

---

## 2 Scope, Status, and Non-Claims

Before introducing the formal structure, it is essential to define clearly what this manuscript does and does not claim.

### This paper **claims** the following:

1. Biological dynamics can be represented in a reduced triadic state space for the purpose of stability analysis.
2. A spiral-time coordinate can be used as an effective embedding variable for time-resolved observables.
3. A residual functional ΔΦ(t) can be defined and used as an operational instability observable.
4. Cellular and DNA-associated dynamics can be treated within the same effective modeling language, provided that the analysis is restricted to time-dependent observables rather than static sequence structure.
5. The framework admits falsification through synthetic simulations, surrogate controls, and experimental time-series validation.

### This paper **does not claim** the following:

1. The discovery of new biological forces.
2. A replacement of established theories, including molecular biology, thermodynamics, statistical mechanics, or systems biology.
3. Direct modeling of static DNA sequence structure as a spiral-time object.
4. Immediate therapeutic efficacy or clinically validated intervention protocols.
5. Evidence for "healing chambers" or other intervention devices.

Accordingly, the present work should be understood as a **conservative, phenomenological effective modeling framework**. Its purpose is to establish a rigorous conceptual and mathematical scaffold that can be systematically tested, challenged, refined, or rejected through simulation and experimental validation.

---

## 3 Triadic State Representation

We introduce a reduced biological state vector:

> **X(t) = ( E(t), I(t), C(t) )** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 1)*

where:

- **E(t)** — an effective *energy-related* observable, such as metabolic flux, ATP/ADP ratio, NADH/FAD dynamics, mitochondrial membrane potential, or other suitable energetic descriptors.
- **I(t)** — an *information-related* observable, such as entropy, regulatory diversity, trajectory complexity, transcriptional heterogeneity, or other coarse-grained measures of state diversity in the broad information-theoretic sense introduced by Shannon [3].
- **C(t)** — a *coherence-related* observable, such as temporal synchronization, spatial phase alignment, oscillatory coordination, or collective mode locking, in line with standard phase-synchrony-based approaches to multivariate signal analysis [8].

The use of a triadic state vector is motivated by the empirical observation that many biological regime transitions involve **simultaneous shifts in energetic state, informational structure, and coherence**. A single-variable biomarker is often insufficient to capture such coupled transitions.

Geometrically, the ICE stability triangle can be interpreted as a **normalized state simplex** in which biological system states are represented by barycentric coordinates X(t) = (E, I, C). For this interpretation to be strictly valid, the components may be mapped onto a normalized domain (e.g., via affine scaling such that E + I + C = 1).

This representation is closely related to the **probability simplex** widely used in information geometry, evolutionary dynamics, and compositional data analysis. Within this interpretation, biological stability corresponds to proximity to a reference equilibrium point X\* inside the simplex, while regime transitions appear as trajectories that move away from this stability region.

> **Definition 1 (ICE state space).** *The ICE state space is the set of admissible state vectors X(t) defined by Eq. (1), together with experimentally specified observables and admissible normalization rules for E, I, and C.*

### Instability Magnitude and Phase Representation

The **instability magnitude** is defined as the Euclidean norm of the deviation vector in ICE space:

> **ΔΦ(t) = √( ΔE(t)² + ΔI(t)² + ΔC(t)² )** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 2)*

This quantity measures the total distance of the system from its reference state, independently of directional weighting.

### Adaptive Phase Definition

To capture directional sensitivity within the ICE manifold, we define a phase-like observable as an adaptively weighted complement of the normalized deviations:

> **φ(t) = 1 − Σ_{k∈{E,I,C}} wₖ(t) Δk(t)** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 3)*

where the weights are given by:

> **wₖ(t) = Δk(t) / ( ΔE(t) + ΔI(t) + ΔC(t) )** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 4)*

ensuring that Σₖ wₖ(t) = 1.

This adaptive weighting dynamically emphasizes the dominant instability component, allowing the phase variable to encode the directional structure of deviations in real time.

### Relation Between Magnitude and Phase

The quantities ΔΦ(t) and φ(t) serve complementary roles: **ΔΦ(t) quantifies the overall magnitude of instability**, while **φ(t) characterizes its directional distribution** within the ICE manifold. Together, this dual representation provides a minimal yet sufficient description of system deviations, enabling the classification of distinct regimes of biological stability.

### Cumulative Instability

In addition to the instantaneous measure, we define a **cumulative instability functional**:

> **J(t) = ∫₀ᵗ ΔΦ(τ) dτ** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 5)*

which captures the accumulated deviation from the reference state over time. This separation is directly analogous to standard distinctions in dynamical systems between instantaneous state deviation and accumulated trajectory cost.

---

## Figure 2 — ICE Stability Triangle

```
                        C
                       /|\
                      / | \
                     /  |  \
                    /   |   \
                   /    |    \
                  /   X(t)   \
                 /    •←ΔΦ    \
                /    ↗         \
               /  X*            \
              /  (reference)     \
             /____________________\
            E                      I

  X* = reference equilibrium state
  ΔΦ = distance from X* to current state X(t)
  Stable: X(t) near X*  →  ΔΦ small
  Unstable: X(t) far from X*  →  ΔΦ large
```

*The distance ΔΦ(t) indicates deviation from a reference state X\*.*

---

## 4 Spiral-Time Embedding

To represent the temporal organization of the triadic state introduced above, we define a structured **spiral-time coordinate**:

> **Ψ(t) = t + iφ(t) + jχ(t)** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 6)*

where:
- **t** is the ordinary forward time coordinate,
- **φ(t)** denotes a phase-memory component,
- **χ(t)** denotes a coherence-bandwidth or memory-depth component.

The purpose of Eq. (6) is not to postulate a literal extension of biological time. Instead, Ψ(t) is introduced as a **structured embedding coordinate** for time-resolved observables. Its role is therefore analogous to an extended state coordinate used in nonlinear dynamics, delay systems, and effective open-system descriptions, rather than a claim about microscopic biological ontology.

> **Remark 1.** *In the present conservative interpretation, φ(t) and χ(t) should be understood as effective descriptors of delayed coupling, phase-history structure, and memory-like dynamical effects already familiar from reduced models of non-Markovian and delayed systems.*

A key point is that the memory component χ(t) should **not** be interpreted as a simple statistical autocorrelation of the signal. A convenient representation can be introduced through a **temporal memory kernel**:

> **χ(t) = ∫₀ᵗ K(t − τ) φ(τ) dτ** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 7)*

where K(t − τ) is a memory kernel and φ(t) denotes the instantaneous phase component of the effective dynamics. In this form, χ(t) accumulates weighted past phase information, with the kernel K determining how strongly earlier states continue to influence the present one.

Equation (7) makes explicit that the proposed spiral-time embedding is compatible with the mathematical structure of **generalized Langevin equations** and, more broadly, with non-Markovian stochastic and open-system models.

---

## Figure 3 — Spiral-Time Embedding Geometry

```
        χ(t) ↑  (coherence bandwidth / memory depth)
              │
              │          ╭────────────────╮
              │         ╱  Ψ(t) trajectory ╲
              │        ╱   spiraling through ╲
              │       ╱    (t, φ, χ) space    ╲
              │      ╱                          ╲
              │─────╱──────────────────────────╱──────→ φ(t)
              │    ╱  (phase-memory component) ╱
              │   ╱                           ╱
              │  ╱                           ╱
              │ ╱                           ╱
              │╱_________________________/
              ╰────────────────────────────────→ t (forward time)

  Ψ(t) = t + iφ(t) + jχ(t)
  • t axis: ordinary temporal progression
  • φ axis: phase-history structure
  • χ axis: accumulated memory effects
```

*Time-resolved biological trajectories are represented as curves in a coordinate system combining forward time, phase-memory structure, and coherence bandwidth. The figure is schematic and intended only to visualize the effective embedding geometry defined in Eq. (6).*

---

### 4.1 Relation to Non-Markovian Dynamics and Projection Formalisms

The representation introduced above can be placed within the broader mathematical context of **reduced non-Markovian dynamics**, where memory effects and deviations from Markovian evolution arise naturally in effective descriptions of complex systems [9].

A typical **generalized Langevin equation (GLE)** can be written as:

> **dx(t)/dt = F(x(t)) + ∫₀ᵗ M(t − τ) x(τ) dτ + η(t)** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 8)*

where M(t − τ) is a memory kernel and η(t) represents an effective stochastic forcing term arising from unresolved environmental degrees of freedom.

Such equations arise naturally in the **Mori–Zwanzig projection formalism**, where a high-dimensional microscopic system is projected onto a reduced set of slow variables. The resulting effective dynamics contain two characteristic terms: a memory kernel describing delayed feedback from eliminated degrees of freedom, and a noise term representing unresolved fluctuations.

Within this perspective, the spiral-time coordinate Ψ(t) = t + iφ(t) + jχ(t) may be interpreted as a structured embedding that separates three aspects of the effective dynamics:

- **t** — forward temporal evolution,
- **φ(t)** — phase-history structure,
- **χ(t)** — accumulated memory effects.

### 4.2 Geometric Operator Form on Trajectories in the ICE Manifold

Let:

> **M_ICE ⊂ ℝ³** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 9)*

denote the reduced state manifold with local coordinates X(t) = ( E(t), I(t), C(t) ). A biological realization then defines a time-parametrized trajectory:

> **γ : [0, T] → M_ICE,   γ(t) = X(t)** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 11)*

For any sufficiently regular scalar observable f : M_ICE → ℝ, we define the **spiral-time trajectory operator** by:

> **D_Ψ f(γ(t)) := d/dt f(γ(t)) + i φ̇(t) v_φ(t) · ∇f(γ(t)) + j χ̇(t) v_χ(t) · ∇f(γ(t))** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 12)*

where ∇f denotes the gradient on M_ICE, and the terms associated with φ̇(t) and χ̇(t) represent effective directional contributions induced by phase-history and memory-like deformations of the trajectory.

Equation (12) should be interpreted conservatively: D_Ψ is not introduced as a microscopic biological law, but as a structured operator acting on trajectory observables in the reduced manifold description. Its purpose is to separate three distinct contributions:

1. the ordinary temporal drift along the trajectory,
2. a phase-history deformation encoded by φ(t),
3. a memory-bandwidth deformation encoded by χ(t).

In the limiting case φ̇(t) → 0, χ̇(t) → 0, the operator reduces to the ordinary directional derivative:

> **D_Ψ f(γ(t)) → d/dt f(γ(t))** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 14)*

### 4.3 Lyapunov-Like Interpretation of the Instability Magnitude

Define:

> **V(t) := ½ ΔΦ(t)²** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 15)*

By construction, V(t) ≥ 0, and V(t) = 0 if and only if ΔE = ΔI = ΔC = 0.

The temporal derivative along a trajectory is given by:

> **dV/dt = ΔΦ(t) · d/dt ΔΦ(t)** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 16)*

If the effective dynamics satisfy dV/dt ≤ 0 in a neighborhood of the reference state, then V(t) is non-increasing. In this regime, ΔΦ(t) acts as a local stability indicator:

- **Decreasing ΔΦ(t)** → motion toward the reference stability region (Isostasis)
- **Increasing ΔΦ(t)** → departure from reference stability region

This interpretation remains purely geometric and does not imply the existence of a global Lyapunov function for the underlying biological system.

---

## 5 Regime Transition Structure

Many biological systems exhibit transitions between stable, metastable, and unstable dynamical regimes. Within the present framework, such transitions are represented as changes in the behavior of the instability magnitude ΔΦ(t) relative to a reference regime.

Operationally, if a system-specific critical value **Φc** is identified empirically, the following qualitative regimes can be distinguished:

| Regime | Condition | Interpretation |
|---|---|---|
| **Stable** | ΔΦ(t) < Φc | Proximity to reference stability region |
| **Metastable** | ΔΦ(t) ≈ Φc | Increased sensitivity to perturbations |
| **Unstable** | ΔΦ(t) > Φc | Sustained deviation from reference regime |

The threshold Φc is not assumed to be universal and must be determined empirically for each system under study, for example through baseline characterization, surrogate analysis, or controlled perturbation experiments.

---

## Figure 4 — Regime Transition Diagram

```
  Stability
  ↑
  │
  │  ████████████
  │  █  STABLE █  ░░░░░░░░░░░░░   ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
  │  ████████████  ░ METASTABLE░   ▒    UNSTABLE     ▒
  │               ░░░░░░░░░░░░░   ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
  │
  │─────────────────────────────────────────────────→ ΔΦ
                              ↑
                              Φc (system-dependent threshold)

  ΔΦ increasing  →  progressive loss of stability
```

*The diagram is not intended to define a universal phase boundary, but rather to provide an effective representation of regime transitions in the reduced state space.*

---

## 6 Theoretical Translation to Cellular Systems

The abstract ICE representation becomes operationally relevant once concrete observables are associated with the three components E, I, and C. Information-related observables may include entropy-based and complexity-based descriptors of time-resolved biological signals, including symbolic or ordinal measures such as permutation entropy [5].

**Table 1: Examples of ICE Observables for Cellular Systems**

| Component | Observable | Example Measurement Modality |
|---|---|---|
| Energy | ATP/ADP ratio | Metabolic assays, luminescence-based reporters |
| Energy | Mitochondrial membrane potential | Fluorescent potential probes |
| Energy | NADH/FAD redox dynamics | Autofluorescence or optical metabolic imaging |
| Information | Transcription entropy | RNA-seq variability, single-cell transcriptomics |
| Information | Gene-regulatory diversity | State-space diversity metrics, burst heterogeneity |
| Information | Trajectory complexity | Entropy rate, symbolic complexity, manifold dispersion |
| Coherence | Calcium synchrony | Calcium imaging |
| Coherence | Cell migration alignment | Wound-healing assays, collective migration analysis |
| Coherence | Morphodynamic coordination | Live-cell imaging, shape covariance |

The resulting cellular trajectory in ICE state space is denoted:

> **X_cell(t) = ( E_cell(t), I_cell(t), C_cell(t) )** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 17)*

To ensure comparability between heterogeneous observables, each component is typically normalized according to:

> **X̃ₖ(t) = ( Xₖ(t) − μₖ ) / σₖ,   k ∈ {E, I, C}** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 18)*

where μₖ and σₖ denote the mean and standard deviation computed over a reference baseline period or control dataset.

This formulation enables comparison across different cellular conditions, including:
- healthy baseline dynamics,
- metabolic stress,
- DNA damage response,
- cellular recovery trajectories.

---

## 7 Theoretical Translation to DNA-Associated Dynamics

A key limitation should be noted: **static DNA sequences are not modeled directly** by the present framework. A nucleotide sequence constitutes a static informational object rather than a time-resolved signal. Consequently, a static sequence cannot be directly embedded in the spiral-time representation of Eq. (6) without introducing additional dynamical observables.

The framework instead targets **DNA-associated dynamical processes**, which naturally generate time-resolved signals. Relevant experimental modalities include time-resolved chromatin dynamics measurements, FRET experiments, live-cell transcription burst recordings, and DNA repair focus tracking.

Examples of DNA-associated dynamical processes include:
- chromatin conformational dynamics,
- transcriptional bursting,
- DNA repair focus dynamics,
- replication-stress trajectories,
- histone-mark switching or epigenetic state transitions,
- locus mobility and nuclear reorganization following perturbation.

Replication-stress-associated state changes may depend on chromatin-bound regulatory layers, including noncoding RNA mechanisms that support fork signaling and genome-maintenance responses [12]. Fork stability depends on active nuclear mechanical support, as nuclear myosin VI contributes directly to the stabilization of stalled or reversed replication forks under stress [19]. Repair-associated chromatin dynamics are also linked to regulated heterochromatin assembly, as the CHAMP1 complex promotes heterochromatin organization together with homology-directed DNA repair [20]. Damage-associated state transitions may further involve large-scale reorganization of genome architecture, consistent with recent evidence that UV-induced changes in 3D genome organization participate directly in the DNA damage response [15].

**Table 2: Examples of DNA-Associated Dynamic Observables**

| Category | Observable | Example Measurement Modality |
|---|---|---|
| Energy-linked | ATP dependence of repair dynamics | Live-cell perturbation, metabolic manipulation |
| Information-linked | Transcription burst variability | Single-cell transcription traces |
| Information-linked | Chromatin accessibility heterogeneity | ATAC-based time-series, chromatin reporters |
| Coherence-linked | Repair focus synchronization | Fluorescence tracking of repair foci |
| Coherence-linked | Locus co-motion | Live-cell microscopy, trajectory correlation |
| Stress-linked | Replication stress markers | Time-dependent marker intensity trajectories |

Using these observables, a DNA-associated state trajectory is written as:

> **X_DNA(t) = ( E_DNA(t), I_DNA(t), C_DNA(t) )** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 19)*

After normalization and embedding in the ICE manifold, the instability magnitude:

> **ΔΦ_DNA(t)** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 20)*

quantifies the Euclidean distance of the current chromatin or DNA-associated dynamical state from a reference regime corresponding to stable cellular operation.

The central hypothesis is therefore that **persistent excursions in ΔΦ_DNA(t) may act as early indicators** of transitions associated with cellular stress, DNA damage response, or repair dynamics.

---

## Figure 5 — DNA-Associated Dynamical Observables

```
  DNA-associated dynamics: conformation, repair, transcription
  ─────────────────────────────────────────────────────────────→ Time

         Burst          Repair Focus
           ●                ●
          ╱ ╲              ╱ ╲
         ╱   ╲            ╱   ╲
        ╱     ╲          ╱     ╲
  ─────╱       ╲────────╱       ╲──────────────────
                                          ●
                                      State Shift
                                         ╱╲
                                        ╱  ╲
  ──────────────────────────────────── ╱    ╲────────

  Legend:
  • Transcription burst  →  structured excursion in I(t)
  • Repair focus         →  coordinated excursion in C(t)
  • State shift          →  transition in E(t), I(t), C(t)
  All events → increase in ΔΦ_DNA(t)
```

*The framework does not model static DNA sequence directly; instead it targets time-resolved processes such as transcription bursts, chromatin reconfiguration, and repair focus dynamics.*

---

## 8 Simulation Framework

The instability magnitude ΔΦ(t) is interpreted as a geometric distance in a reduced state space and does not imply a variational or functional formalism. Synthetic simulations constitute the **first falsification layer** of the present framework.

### 8.1 Simulation Goals

The simulation program is designed to address the following four questions:

1. Can a triadic state representation (E, I, C) be reliably reconstructed from synthetic multivariate dynamics in a stable and interpretable manner?
2. Does the instability magnitude ΔΦ(t) exhibit a systematic increase prior to or during known regime transitions?
3. Is the behavior of ΔΦ(t) robust under moderate noise levels and parameter perturbations?
4. Does the instability magnitude provide additional discriminative power compared to simpler baseline measures under matched surrogate conditions?

### 8.2 Candidate Synthetic Systems

Suitable example classes include:
- respiratory oscillator models with progressive desynchronization,
- coupled calcium-signaling oscillators,
- stochastic transcription-burst processes,
- repair-and-failure switching models,
- delayed nonlinear oscillators with memory effects.

### 8.3 Minimal Synthetic Validation of the ΔΦ Operator

**Synthetic signal construction.** A univariate time series s(t) is sampled at f_s = 50 Hz over T = 180 s with three distinct regimes:

**Stable regime** (t ∈ [0, 60) s):

> **s(t) = sin(2π · 0.25 · t) + 𝒩(0, 0.05)** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 21)*

**Transition regime** (t ∈ [60, 120) s), with slow stochastic drift:

> **s(t) = (1 + η_A(t)) sin( 2π(0.25 + η_f(t))t ) + 𝒩(0, 0.2)** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 22)*

where η_A(t) and η_f(t) are low-frequency random-walk processes.

**Unstable regime** (t ∈ [120, 180] s), with strong stochastic frequency modulation:

> **s(t) = sin(2πf(t)t) + 0.8 𝒩(0, 1),   f(t) ~ 𝒰(0.1, 0.6)** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 23)*

**Feature extraction.** For each sliding window of length W (8 seconds):

> **E(t) = √( (1/W) Σ sₖ² )** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 24)*

> **I(t) = −Σₖ pₖ log pₖ** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 25)*

> **C(t) = 1 / ( Var(Δφ(t)) + ε )** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 26)*

where pₖ denotes the normalized power spectrum and φ(t) is the instantaneous phase obtained via the Hilbert transform.

Baseline statistics (μ_X, σ_X) are computed over the stable regime (t < 60 s), and each feature is standardized:

> **Xₙ(t) = ( X(t) − μ_X ) / σ_X,   X ∈ {E, I, C}** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 27)*

**Instability magnitude and evaluation.** The instability magnitude is:

> **ΔΦ(t) = √( Eₙ(t)² + Iₙ(t)² + Cₙ(t)² )** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 28)*

A valid instability indicator must satisfy:

> **ΔΦ_stable < ΔΦ_transition < ΔΦ_unstable** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 29)*

### 8.4 Minimal Triadic Mapping for Simulation

A minimal proof-of-concept mapping into the ICE state space is:

> **E(t) := A_e(t),   I(t) := H_w(t),   C(t) := R(t)** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 30)*

where A_e(t) is an amplitude or energy proxy, H_w(t) is a windowed entropy or complexity measure, and R(t) is a coherence or phase-locking measure.

For a respiratory model specifically:

> **E(t) := A(t),   I(t) := H_w(t),   C(t) := R(t)** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 31)*

where A(t) is the respiratory envelope amplitude, H_w(t) is a windowed entropy measure, and R(t) quantifies inter-cycle regularity or phase consistency.

### 8.5 Reference State and Instability Magnitude

A reference stable state is defined using a baseline window [t₀, t₁]:

> **x\* = (0, 0, 0)** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 32)*

The instability magnitude is then:

> **ΔΦ(t) = √( Eₙ(t)² + Iₙ(t)² + Cₙ(t)² )** &nbsp;&nbsp;&nbsp;&nbsp;*(Eq. 33)*

### 8.6 Expected Simulation Behavior

For a successful proof-of-concept:

1. During initial stable regime: E(t), I(t), C(t) remain close to baseline; ΔΦ(t) remains low and approximately stationary.
2. As control parameter approaches transition: at least one triadic component begins systematic drift away from baseline.
3. Near transition boundary: combined deviation produces a clear and sustained increase in ΔΦ(t).
4. After entering unstable regime: ΔΦ(t) remains elevated or exhibits burst-like excursions.

### 8.7 Simulation Result Panels

---

## Figure 7 — Controlled Triadic Instability Scenario

```
  ┌─────────────────────────────────────────────────────────────────┐
  │ Panel 1: Raw Synthetic Signal  s(t)                             │
  │  3 ┤                                                            │
  │  2 ┤  ╭╮ ╭╮ ╭╮ ╭╮ ╭╮ ╭╮ ╭╮│    ╭╮╭╮╭╮╭╮│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
  │  1 ┤ ╭╯╰─╯╰─╯╰─╯╰─╯╰─╯╰─╯╰│╭─╮╭╯╰╯╰╯╰╯╰│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
  │  0 ┤─────────────────────────│────────────│────────────────────  │
  │ -1 ┤ ╰─╮╭─╮╭─╮╭─╮╭─╮╭─╮   │╰──╯        │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
  │ -2 ┤   ╰╯  ╰╯  ╰╯  ╰╯  ╰╯  │            │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
  │ -3 ┤                        │            │                      │
  │    └────────────────────────┴────────────┴──────────────────── │
  │         0        25    50   60    75   100  120   150      175  │
  │                  ←─STABLE─→  ←─PRE-INST─→  ←──UNSTABLE──────→  │
  │                              ↑t=60s         ↑t=120s             │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │ Panel 2: Baseline-Normalized Triadic Features Eₙ(t), Iₙ(t), Cₙ(t)│
  │ 20 ┤                                        ╭──E(t)             │
  │ 15 ┤                                       ╱│╲  ╭╮ ╭╮ ╭╮        │
  │ 10 ┤                              ╭──────╮╱ │ ╲╭╯╰─╯╰─╯╰        │
  │  5 ┤                        ╭────╯      ╰╯  │  I(t) ~           │
  │  0 ┤────────────────────────┼───────────────┼────────── C(t) ─── │
  │ -5 ┤                        │               │                    │
  │    └────────────────────────┴───────────────┴─────────────────── │
  │         0        25    50   60    75   100  120   150      175   │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │ Panel 3: Instability Magnitude ΔΦ(t)                            │
  │500 ┤                                        ╭──╮  ╭╮  ╭╮        │
  │400 ┤                                       ╱│  ╰──╯╰──╯╰        │
  │300 ┤                                      ╱ │                    │
  │200 ┤                              ╭──────╯  │                    │
  │100 ┤                      ╭──────╯          │                    │
  │  0 ┤──────────────────────┼─────────────────┼──────────────────  │
  │    └──────────────────────┴─────────────────┴──────────────────  │
  │         0        25    50   60    75   100  120   150      175   │
  │                            ↑t=60s            ↑t=120s            │
  └─────────────────────────────────────────────────────────────────┘
```

*Top: Raw synthetic signal with predefined regime boundaries. Second: Baseline-normalized triadic observables Eₙ(t), Iₙ(t), and Cₙ(t) exhibiting coordinated but non-identical evolution. Third: Instability magnitude ΔΦ(t) showing a structured increase preceding the transition.*

---

### 8.8 Simulation Summary Table

**Table 3: Simulation Performance Summary (Respiratory Synthetic System)**

| Metric | Value | Std | Notes |
|---|---|---|---|
| Lead time (s) | 109.14 | 4.94 | Detection before transition onset t₂ |
| AUC (ΔΦ) | 0.9408 | — | Triadic model |
| AUC (E only) | 0.8498 | — | Baseline |
| AUC (I only) | 0.8610 | — | Baseline |
| AUC (C only) | 0.8412 | — | Baseline |
| Ordering success (%) | 100.0 | — | stable < pre < unstable |

*Note: The lead-time statistic is reported from the Stage 3 multi-seed benchmark, whereas the AUC values are computed from the deterministic respiratory synthetic benchmark. These results arise from different synthetic protocols and should not be interpreted as originating from a single identical signal generator.*

The triadic instability magnitude ΔΦ(t) achieved an AUC of **0.9408**, exceeding all single-component baselines (AUC_E = 0.8498, AUC_I = 0.8610, AUC_C = 0.8412). The regime-ordering criterion ΔΦ_stable < ΔΦ_pre < ΔΦ_unstable was satisfied in **100.0%** of evaluated runs.

### 8.9 Recommended First Implementation

For the first practical implementation, the following minimal pipeline is recommended:

1. respiratory oscillator simulation,
2. extraction of amplitude, entropy, and phase-regularity proxies,
3. computation of E(t), I(t), C(t) via Eq. (31),
4. evaluation of ΔΦ(t) via Eq. (28),
5. comparison to surrogate and shuffled controls.

---

## Figure 6 — Simulation Pipeline

```
  ┌────────────────┐     ┌──────────────────┐     ┌───────────────┐
  │  Synthetic     │────▶│  Multivariate     │────▶│  ICE mapping  │
  │  model         │     │  signals          │     │  (E, I, C)    │
  └────────────────┘     └──────────────────┘     └───────┬───────┘
                                                          │
                                                          ▼
                         ┌──────────────────┐     ┌───────────────┐
                         │  Transition       │◀────│  Instability  │
                         │  evaluation       │     │  ΔΦ(t)        │
                         └──────────────────┘     └───────────────┘
```

*Synthetic multivariate signals are mapped into the ICE state space, from which the instability magnitude ΔΦ(t) is computed and evaluated against known transition structure.*

---

## 9 Null Tests and Baseline Controls

Null tests are not an optional supplement, but a **central component** of the proposed methodology. Such controls are essential to distinguish genuine transition-related instability from generic stochastic variability [7].

### 9.1 Null-Test Classes

At minimum, the following control classes should be implemented:

1. **Phase-randomized surrogate controls:** preserve marginal spectra while destroying structured phase relations.
2. **Shuffled trajectory controls:** destroy temporal ordering while preserving value distributions.
3. **Matched nonlinear baselines:** compare ΔΦ(t) against established nonlinear indicators such as entropy rate, delay-embedding instability measures, or Lyapunov-style surrogates.
4. **False-positive controls:** evaluate stable synthetic systems with no true transition structure.

### 9.2 Expected Null-Test Outcome

| Condition | AUC | Interpretation |
|---|---|---|
| Observed synthetic benchmark | 0.9408 | Strong discrimination |
| Phase-randomized surrogate | 0.6366 | Partial attenuation |
| Shuffled-time null | 0.4945 | Near-chance (temporal ordering essential) |
| Stable control | 0.4937 | No spurious detection |

---

## Figure 8 — Null-Test Comparison

```
  ΔΦ(t)
   25 ┤
      │  ░░░ Shuffled-time null     ─── Stable control
      │  ▒▒▒ Phase-randomized null  ███ Observed
   20 ┤
      │  ████     ████         ████   ████    ████
   15 ┤  ████  ████████     ██████████████ ████████
      │  ████████████████  ███████████████████████
   10 ┤  █████████████████████████████████████████
      │  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
    5 ┤  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
      │  ─────────────────────────────────────────  ← Stable control
    0 ┤
      └───────────────────────────────────────────→ Time (s)
        0       20      40      60      80     100     120    140    160

  AUC values:
  ▓ Observed:              0.9408  ← strong discrimination
  ▒ Phase-randomized null: 0.6366  ← partial attenuation
  ░ Shuffled-time null:    0.4945  ← near chance
  ─ Stable control:        0.4937  ← no spurious detection
```

*The observed signal shows structured transition-related excursions that are not retained under temporal shuffling and are substantially attenuated under phase randomization, while the stable control remains uniformly low.*

---

## 10 Interpretation and Limits

The present framework is best interpreted as a **reduced stability language** for time-resolved biological systems.

Several limitations must be explicitly acknowledged:

1. **The ICE variables are coarse-grained and context-dependent.** They do not represent fundamental observables and may admit multiple non-equivalent experimental realizations.

2. **Ψ(t) should be understood as a structured embedding coordinate**, not as an established ontological description of biological time.

3. **ΔΦ(t) is defined relative to a chosen reference state or trajectory.** Its interpretation is therefore intrinsically relational and not absolute.

4. **No universal threshold is implied.** Any threshold used in a specific application must be estimated and validated within the corresponding system and dataset.

5. **The framework is not a therapeutic model.** It does not establish causal intervention mechanisms and should not be interpreted as supporting claims about medical devices, chambers, or field-based treatment systems.

These limitations are not deficiencies, but **necessary conditions for methodological rigor and falsifiability**.

---

## 11 Discussion

Despite its deliberately conservative scope, the proposed framework provides a unified perspective for the analysis of time-resolved biological dynamics. Within this representation, the instability magnitude ΔΦ(t) acts as a coarse-grained measure of deviation from a reference stability manifold defined in the triadic ICE state space.

Two potential advantages follow from this formulation:

**First**, the framework enables **cross-domain comparison**. Processes such as respiratory instability, neural desynchronization, calcium signaling transitions, or DNA-repair failures are mechanistically distinct. However, each can manifest as a transition between relatively stable and unstable regions of the reduced state space. The ICE representation therefore provides a common coordinate system in which such transitions can be compared at the level of dynamical structure rather than biochemical detail.

**Second**, the framework defines a mathematically explicit structure that may support future model-guided analysis and control strategies. If future experimental studies demonstrate that controlled perturbations consistently reduce ΔΦ(t) in a reproducible manner, the framework could serve as a basis for investigating closed-loop stabilization or adaptive control schemes. Such applications remain speculative and lie beyond the scope of the present study.

At the current stage, the **primary contribution is methodological** rather than mechanistic. The framework provides a rigorous and falsifiable structure for multivariate regime-transition analysis in biological systems.

---

## 12 Outlook

Future work should proceed through a staged validation program:

### Stage I: Synthetic Validation
Systematic falsification using controlled synthetic systems. Simulation environments representing respiratory oscillators, calcium-signaling networks, and simplified DNA-repair dynamics allow evaluation of whether ΔΦ(t) reliably signals regime transitions under known conditions, while remaining robust to noise, parameter variation, and surrogate controls.

### Stage II: In Vitro Validation
Candidate datasets include:
- wound-healing assays tracking collective cell migration,
- calcium imaging under controlled perturbations,
- chromatin mobility measurements in live-cell microscopy,
- live-cell DNA repair focus tracking,
- replication-stress and recovery trajectories.

### Stage III: Controlled Perturbation Studies
Only after robust empirical validation would it become scientifically meaningful to investigate whether controlled perturbations can reproducibly shift biological systems toward lower-ΔΦ regimes. Such studies require carefully designed feedback protocols and must be interpreted strictly within established biophysical constraints.

---

## Funding

This research received no external funding.

## Data Availability

No experimental datasets were generated in this study. The present manuscript is theoretical and methodological in scope. Synthetic simulation code, figure-generation scripts, and supplementary reproducibility materials are available in the public GitHub repository: [https://github.com/dfeen87/Cell-DNA-Dynamics](https://github.com/dfeen87/Cell-DNA-Dynamics)

## Conflict of Interest

The authors declare no conflict of interest.

## Author Contributions

**Marcel Krüger** conceived the central research idea and developed the theoretical framework underlying this work. M. Krüger designed the mathematical structure of the model, including the instability functional ΔΦ, and developed the conceptual extension of the framework to cellular and DNA-associated dynamical processes.

**Don Michael Feeney Jr.** contributed to the computational and simulation aspects of the project, including the design and implementation of the synthetic simulation pipelines used for proof-of-concept validation. He also generated the manuscript simulation figures, reviewed the manuscript, provided technical feedback, and contributed to refining the presentation and final structure of the paper.

Both authors discussed the results, contributed to the final manuscript preparation, and approved the submitted version.

## AI Assistance Statement

During the preparation of this manuscript, AI-based language assistance was used for drafting support, language polishing, and LaTeX structuring. All scientific ideas, conceptual decisions, mathematical definitions, scope limitations, and interpretations were reviewed and approved by the human authors, who take full responsibility for the content.

---

## References

[1] Krüger, M., Wende, M. T., & Feeney, D. (2026). Information-Theoretic Modeling of Neural Coherence via Triadic Spiral-Time Dynamics: A Framework for Neurodynamic Collapse. *Zenodo*. https://doi.org/10.5281/zenodo.18867962

[2] Krüger, M., & Feeney, D. (2026). A Spiral-Time Operator Framework for Heart–Brain Isostasis: Operator-Based Early Detection of Cardiac Instability. *Zenodo*. https://doi.org/10.5281/zenodo.18640469

[3] Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27, 379–423, 623–656. doi:10.1002/j.1538-7305.1948.tb01338.x

[4] Kantz, H., & Schreiber, T. (2004). *Nonlinear Time Series Analysis* (2nd ed.). Cambridge University Press.

[5] Bandt, C., & Pompe, B. (2002). Permutation entropy: A natural complexity measure for time series. *Physical Review Letters*, 88, 174102. doi:10.1103/PhysRevLett.88.174102

[6] Scheffer, M., Bascompte, J., Brock, W. A., et al. (2009). Early-warning signals for critical transitions. *Nature*, 461, 53–59. doi:10.1038/nature08227

[7] Dakos, V., Carpenter, S. R., Brock, W. A., et al. (2012). Methods for detecting early warnings of critical transitions in time series illustrated using simulated ecological data. *PLoS ONE*, 7(7), e41010. doi:10.1371/journal.pone.0041010

[8] Lachaux, J. P., Rodriguez, E., Martinerie, J., & Varela, F. J. (1999). Measuring phase synchrony in brain signals. *Human Brain Mapping*, 8(4), 194–208. doi:10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C

[9] Rivas, Á., Huelga, S. F., & Plenio, M. B. (2014). Quantum non-Markovianity: characterization, quantification and detection. *Reports on Progress in Physics*, 77, 094001. doi:10.1088/0034-4885/77/9/094001

[10] Milz, S., Pollock, F. A., & Modi, K. (2017). An introduction to operational quantum dynamics. *Open Systems & Information Dynamics*, 24(4), 1740016. doi:10.1142/S1230161217400169

[11] Idesis, S., Allegra, M., Vohrizek, J., et al. (2024). Generative whole-brain dynamics models from healthy subjects predict functional alterations in stroke at the level of individual patients. *Brain Communications*, 6(4), fcae237. doi:10.1093/braincomms/fcae237

[12] Montes, M., Ranieri, A., Goñi, E., et al. (2024). The chromatin-associated lncREST ensures effective replication stress response by promoting the assembly of fork signaling factors. *Nature Communications*, 15(1), 978. doi:10.1038/s41467-024-45183-5

[13] Gaggioli, V., Lo, C. S. Y., Reverón-Gómez, N., et al. (2023). Dynamic de novo heterochromatin assembly and disassembly at replication forks ensures fork stability. *Nature Cell Biology*, 25(7), 1017–1032. doi:10.1038/s41556-023-01167-z

[14] Chu, F.-Y., Clavijo, A. S., Lee, S., & Zidovska, A. (2024). Transcription-dependent mobility of single genes and genome-wide motions in live human cells. *Nature Communications*, 15, 8879. doi:10.1038/s41467-024-51149-4

[15] Kaya, V. O., & Adebali, O. (2025). UV-induced reorganization of 3D genome mediates DNA damage response. *Nature Communications*, 16(1), 1376. doi:10.1038/s41467-024-55724-7

[16] da Silva, R. C., Eleftheriou, K., Recchia, D. C., et al. (2025). Engineered chromatin readers track damaged chromatin dynamics in live cells and animals. *Nature Communications*, 16(1), 10127. doi:10.1038/s41467-025-65706-y

[17] de Luca, K. L., Rullens, P. M. J., Karpinska, M. A., et al. (2024). Genome-wide profiling of DNA repair proteins in single cells. *Nature Communications*, 15(1), 9918. doi:10.1038/s41467-024-54159-4

[18] Palumbieri, M. D., Merigliano, C., González-Acosta, D., et al. (2023). Nuclear actin polymerization rapidly mediates replication fork remodeling upon stress by limiting PrimPol activity. *Nature Communications*, 14(1), 7819. doi:10.1038/s41467-023-43183-5

[19] Shi, J., Hauschulte, K., Mikicic, I., et al. (2023). Nuclear myosin VI maintains replication fork stability. *Nature Communications*, 14(1), 3787. doi:10.1038/s41467-023-39517-y

[20] Li, F., Zhang, T., Syed, A., et al. (2025). CHAMP1 complex directs heterochromatin assembly and promotes homology-directed DNA repair. *Nature Communications*, 16(1), 1714. doi:10.1038/s41467-025-56834-6
