# VALIDATION_ANALYSIS.md

**Triadic Stability Framework — Scientific Validation Document**
*Cell-DNA-Dynamics Repository · Stages 1–5*

---

## 1. Overview

### Purpose of the Validation Suite

This document summarizes the validation evidence produced by the computational pipeline for the manuscript *"A Triadic Stability Framework for Detecting Regime Transitions in Time-Resolved Biological Systems"* (Krüger and Feeney Jr.). The pipeline operationalizes the triadic Energy–Information–Coherence (ICE) representation and the instability magnitude ΔΦ(t) introduced in the manuscript, and establishes a staged, falsifiable validation protocol based on synthetic simulations, surrogate-data controls, and multi-seed benchmarking.

The framework represents biological dynamics in a reduced state space defined by three scalar observables:

> *X(t) = ( E(t), I(t), C(t) )* — manuscript Eq. (1)

where E(t) is an energy-related observable, I(t) is an information-related observable, and C(t) is a coherence-related observable. The instability magnitude

> *ΔΦ(t) = sqrt( ΔE(t)² + ΔI(t)² + ΔC(t)² )* — manuscript Eq. (2)

quantifies the Euclidean distance of the system state from the reference stability region x* = (0, 0, 0) in the normalised ICE space.

### Linkage to Manuscript Sections

| Document Section | Manuscript Section(s) |
|---|---|
| Stage 1 — Preprocessing & Normalization | §8.3 (Synthetic Validation Protocol), §8.4 (Minimal triadic mapping), §8.5 (Reference state) |
| Stage 2 — φ(t) and χ(t) Construction | §3 (Adaptive phase definition), §4 (Spiral-Time Embedding), §4.1 |
| Stage 3 — Ground-Truth Validation | §8.6–§8.9, §9 (Null Tests) |
| Stage 4 — Embedding & ΔΦ(t) | §4.2 (Geometric operator), §4.3 (Lyapunov interpretation) |
| Stage 5 — Regime Segmentation | §5 (Regime Transition Structure), §6 (Cellular Systems) |
| Table 3 Alignment | §8.8 (Simulation summary table) |

### Summary of Stages 1–5

| Stage | Module | Primary Output |
|---|---|---|
| Stage 1 | `src/stage1/stage_i.py` | `data/stage1/stage_i_results.csv`, `figures/stage1/stage_i_figure.png` |
| Stage 2 | `src/stage2/stage_ii.py` | `data/stage2/stage_ii_segmentation.csv`, `figures/stage2/stage_ii_regimes.png` |
| Stage 3 | `src/stage3/stage_iii.py` | `data/stage3/stage_iii_summary.csv`, `figures/stage3/stage_iii_seed0.png` |
| Stage 4 | `src/stage4_spiral_time/` | `figures/stage4_spiral_time/phi_chi_curves.png`, `figures/stage4_spiral_time/spiral_embedding.png` |
| Stage 5 | `src/stage5_regime_transitions/` | `figures/stage5_regime_transitions/delta_phi_thresholds.png`, `figures/stage5_regime_transitions/regime_segmentation.png` |

---

## 2. Stage 1 — Preprocessing & Normalization

### Input Data Sources

Stage 1 operates entirely on a deterministic synthetic signal constructed in `src/stage1/stage_i.py` according to the protocol specified in manuscript §8.3. No external experimental data are required. The signal is sampled at f_s = 50 Hz over a total duration of T = 180 s (N = 9 000 samples), giving a time vector t ∈ [0, 180) s.

**Signal construction** follows the three-regime structure of manuscript §8.3:

- *Stable regime* (t ∈ [0, 60) s): harmonic oscillation — `s(t) = sin(2π · 0.25 · t) + N(0, 0.05)` — Eq. (21)
- *Transition regime* (t ∈ [60, 120) s): slow stochastic amplitude and frequency drift — `s(t) = (1 + 0.5 η_A) sin(2π(0.25 + 0.1 η_f)t) + N(0, 0.2)` — Eq. (22), where η_A and η_f are Gaussian random walks
- *Unstable regime* (t ∈ [120, 180] s): strong stochastic frequency modulation — `s(t) = sin(2πf(t)t) + 0.8 N(0,1)`, f(t) ~ U(0.1, 0.6) — Eq. (23)

All random processes use `numpy.random.default_rng(seed=42)`.

### Cleaning Steps

No missing-value imputation or outlier removal is applied. Numerical guards are used in feature extraction:

- A small constant ε = 1×10⁻⁶ is added to the denominator of the coherence metric to prevent division by zero.
- Standard deviations are lower-bounded at ε during normalization to prevent degenerate scaling.

### Feature Extraction

Three feature channels are extracted via a sliding window of width W = 400 samples (8 s at 50 Hz), step = 1 sample, producing n_windows = 8 601 windows (time-stamped at window centres, covering approximately t ∈ [4, 176] s):

| Feature | Formula | Interpretation |
|---|---|---|
| E(t) | `sqrt( (1/W) Σ s_k² )` — Eq. (24) | RMS energy proxy |
| I(t) | `−Σ p_k log p_k` — Eq. (25) | Shannon entropy of the normalised power spectrum |
| C(t) | `1 / ( Var(Δφ(t)) + ε )` — Eq. (26) | Phase coherence via Hilbert transform; inverse instantaneous-phase-difference variance |

### Normalization Logic

Baseline statistics are computed over windows whose center falls in the stable regime (t_feat < 60 s), corresponding to manuscript §8.3:

```
μ_X = mean(X[baseline]),  σ_X = std(X[baseline]),  X ∈ {E, I, C}
X_n(t) = ( X(t) − μ_X ) / σ_X     — Eq. (27)
```

After normalization the reference state is x* = (0, 0, 0) by construction. The instability magnitude is

```
ΔΦ(t) = sqrt( E_n(t)² + I_n(t)² + C_n(t)² )     — Eq. (28)
```

with uniform weights w_E = w_I = w_C = 1.

### Outputs

| Artefact | Path | Description |
|---|---|---|
| Results CSV | `data/stage1/stage_i_results.csv` | 8 601 rows; columns: `time`, `E`, `I`, `C`, `delta_phi` (raw E/I/C feature values and ΔΦ(t)) |
| Figure | `figures/stage1/stage_i_figure.png` | Three-panel figure: (1) raw signal s(t), (2) normalised features E_n(t)/I_n(t)/C_n(t), (3) ΔΦ(t) with regime boundary lines at t = 60 s and t = 120 s |

**Sample data (first three rows of `stage_i_results.csv`):**

```
time,E,I,C,delta_phi
4.0000,0.500881,0.051152,242.612022,1.013564
4.0200,0.500881,0.051161,243.223645,1.035980
4.0400,0.500884,0.050947,244.284397,1.075717
```

**Regime ΔΦ ordering check (Stage 1 internal validation):**
The pipeline prints and verifies the ordering constraint (manuscript Eq. 29):
> ΔΦ_stable < ΔΦ_transition < ΔΦ_unstable

---

## 3. Stage 2 — φ(t) and χ(t) Construction

### Definitions from PAPER.md

Stage 2 implements two complementary operations that are conceptually grounded in the spiral-time embedding defined in manuscript §§3–4.

**Adaptive phase φ(t)** is defined in manuscript Eq. (3)–(4):

```
φ(t) = 1 − Σ_{k ∈ {E,I,C}} w_k(t) Δk(t)     (Eq. 3)
w_k(t) = Δk(t) / ( ΔE(t) + ΔI(t) + ΔC(t) )    (Eq. 4)
```

ensuring Σ_k w_k(t) = 1. φ(t) characterises the *directional distribution* of instability within the ICE manifold.

**Memory component χ(t)** is defined in manuscript Eq. (7):

```
χ(t) = ∫₀ᵗ K(t − τ) φ(τ) dτ     (Eq. 7)
```

where K(t − τ) is a strictly causal memory kernel. χ(t) accumulates weighted past phase information, making explicit the non-Markovian history-dependent structure referenced in §4.1.

### Smoothing Parameters

**Stage 2 regime segmentation** (`src/stage2/stage_ii.py`) applies a Savitzky–Golay filter to ΔΦ(t) before threshold computation:

| Parameter | Value | Description |
|---|---|---|
| `SG_WINDOW` | 251 samples | Window length (must be odd; validated at runtime) |
| `SG_POLYORD` | 3 | Polynomial order |

This smoothing is used for threshold computation and visualisation only; all quantitative evaluation uses the unsmoothed ΔΦ(t).

**Stage 4 φ(t) computation** (`src/stage4_spiral_time/compute_phi.py`) uses baseline-normalised absolute deviations:

```
Δk(t) = |raw_k(t) − μ_k| / (σ_k + ε),  ε = 1e-8
```

computed over the first `baseline_frac = 0.1` (10%) of the series. Weights are clipped to [0, 1] and renormalised.

**Stage 4 χ(t) computation** (`src/stage4_spiral_time/compute_chi.py`) uses a causal convolution with row-wise kernel normalisation. Supported kernels:

- **Exponential:** K(s) = λ exp(−λs), s > 0; default λ = 1.0
- **Gaussian:** K(s) = (1/(σ√(2π))) exp(−s²/(2σ²)), s > 0; default σ = 1.0

χ(t) is standardised to zero mean and unit variance after convolution to provide a stable scale for derivative computation.

### Embedding Logic

The spiral-time coordinate (manuscript Eq. 6) is:

```
Ψ(t) = t + i φ(t) + j χ(t)
```

where t is the ordinary forward time, φ(t) the phase-memory component, and χ(t) the coherence-bandwidth or memory-depth component. This serves as a structured embedding coordinate for time-resolved observables — not as a literal extension of biological time (Remark 1 of the manuscript).

**Stage 2 segmentation logic** reads ΔΦ(t) from Stage 1, applies the Savitzky–Golay filter, and assigns three regime labels using quantile thresholds Q25 and Q75 of the smoothed signal:

```
smoothed ΔΦ < Q25  → "stable"
Q25 ≤ smoothed ΔΦ ≤ Q75  → "pre_instability"
smoothed ΔΦ > Q75  → "instability"
```

### Outputs

| Artefact | Path | Description |
|---|---|---|
| Segmentation CSV | `data/stage2/stage_ii_segmentation.csv` | 8 601 rows; columns: `t`, `regime`, `DeltaPhi`, `E`, `I`, `C` |
| Figure | `figures/stage2/stage_ii_regimes.png` | Three-panel figure with regime shading: (1) raw ΔΦ(t), (2) features E/I/C, (3) smoothed ΔΦ(t) with Q25/Q75 threshold lines |

**Sample data (first three rows of `stage_ii_segmentation.csv`):**

```
t,regime,DeltaPhi,E,I,C
4.0,pre_instability,1.013564,0.500881,0.051152,242.612022
4.02,pre_instability,1.03598,0.500881,0.051161,243.223645
4.04,pre_instability,1.075717,0.500884,0.050947,244.284397
```

---

## 4. Stage 3 — Ground-Truth Validation

### Synthetic Triadic System

Stage 3 (`src/stage3/stage_iii.py`) generates a more detailed synthetic signal with an explicit three-regime ground truth (y_regime ∈ {0, 1, 2} for stable / pre-instability / instability). Signal parameters evolve continuously across regime boundaries:

| Parameter | Stable (t < 60 s) | Pre-instability (60–120 s) | Instability (t ≥ 120 s) |
|---|---|---|---|
| Amplitude A | 1.0 + 0.02 N(0,1) | 1.0 + 0.004τ + 0.03 N | 1.3 + 0.006τ + 0.05 N |
| Frequency f | 0.25 + 0.003 N | 0.25 + 0.0015τ + 0.005 N | 0.35 + 0.03 sin(0.2t) + 0.01 N |
| Noise σ | 0.04 | 0.10 + 0.001τ | 0.30 |
| Burst component | none | 0.15 sin(2π·1.8t) w.p. 0.05 | 0.35 N(0,1) w.p. 0.10 |

Feature extraction uses an 8 s window (400 samples at 50 Hz). Features are z-score normalised against the stable regime (y_regime == 0). ΔΦ(t) is computed as manuscript Eq. (28).

### Early-Warning Behavior

The lead-time threshold is derived from the 90th quantile of ΔΦ restricted to the stable segment (`lead_threshold_quantile = 0.90`). Lead time is computed as:

```
lead_time = t2 − t_first_cross
```

where t_first_cross is the first time before t2 = 120 s at which ΔΦ(t) exceeds the threshold. Across 50 seeds (seed 0–49):

> **Mean lead time: 109.1412 ± 4.9390 s** (ordering_rate = 100.0%)

This demonstrates that ΔΦ(t) crosses the threshold substantially before the ground-truth transition onset, providing an early-warning signal consistent with the expectation described in manuscript §8.6.

### Ablation Tests

Four ΔΦ variants are computed per seed:

| Variant | Formula |
|---|---|
| Full ΔΦ | `sqrt(E_n² + I_n² + C_n²)` |
| ΔΦ_EI | `sqrt(E_n² + I_n²)` |
| ΔΦ_EC | `sqrt(E_n² + C_n²)` |
| ΔΦ_IC | `sqrt(I_n² + C_n²)` |

AUC results across 50 seeds (mean ± std from `data/stage3/stage_iii_summary.csv`):

| Metric | Mean ± Std |
|---|---|
| AUC_full (ΔΦ) | **0.9998 ± 0.0002** |
| AUC_EI | 0.9998 ± 0.0002 |
| AUC_EC | 0.9998 ± 0.0002 |
| AUC_IC | 0.9986 ± 0.0008 |

### Baseline Comparisons

Single-component baselines (|E_n|, |I_n|, |C_n|) are evaluated as AUC scores:

| Metric | Mean ± Std |
|---|---|
| AUC_E_only | 0.9998 ± 0.0002 |
| AUC_I_only | 0.9984 ± 0.0010 |
| AUC_C_only | 0.9997 ± 0.0004 |

The full ΔΦ meets or exceeds each single-component baseline in mean AUC across all 50 seeds, consistent with manuscript §8.1 question 4.

**Regime ΔΦ ordering (mean ± std):**

| Regime | Mean ΔΦ |
|---|---|
| Stable | 1.5829 ± 0.0351 |
| Pre-instability | 48.6255 ± 6.8112 |
| Instability | 168.0447 ± 33.8074 |

The strict ordering ΔΦ_stable < ΔΦ_pre < ΔΦ_instability holds in **100.0% of seeds** (manuscript Eq. 29).

### Null-Test Descriptions

Null tests are implemented in `src/simulation/null_tests.py` and match the control classes specified in manuscript §9.1:

1. **Shuffled-time null** (`data/synthetic/null_tests/shuffled.csv`): All ΔΦ(t) values are retained but their temporal order is randomised by random permutation (seed = 0). This destroys temporal ordering while preserving the value distribution. Expected outcome: regime structure collapses and ordered separation is absent (manuscript §9.2: "Shuffled trajectories: temporal structure is removed").

2. **Phase-randomised surrogate** (`data/synthetic/null_tests/surrogate.csv`): Each ICE component's Fourier phases are replaced with independent uniform-random values in [0, 2π) while the amplitude spectrum is preserved (DC and Nyquist phases fixed for strict real-valued output). ΔΦ(t) is recomputed from the three surrogate components. Expected outcome: "transition signatures are significantly attenuated, as phase coherence is destroyed while spectral content is preserved" (manuscript §9.2).

3. **Stable control** (`data/synthetic/null_tests/stable_control.csv`): A purely stable system is simulated by drawing each ICE component from white noise with σ = 0.05, yielding a low-ΔΦ baseline throughout (manuscript §9.1: "False-positive controls: evaluate stable synthetic systems with no true transition structure"). Expected outcome: "ΔΦ(t) remains low and does not exhibit systematic increases" (manuscript §9.2).

These three controls together isolate phase coherence, temporal ordering, and spurious detections — the minimal sufficient set described in manuscript §9.1. Results are compared to the observed ΔΦ(t) in `simulations/null_tests_comparison.png`.

### Figure Reference

`figures/stage3/stage_iii_seed0.png` shows a four-panel display for seed 0: (1) raw signal with regime shading, (2) normalised features E_n/I_n/C_n, (3) raw ΔΦ(t), (4) smoothed ΔΦ(t) (Savitzky–Golay, window = 101, polyorder = 3; visualisation only).

---

## 5. Stage 4 — Embedding & ΔΦ(t)

### Embedding Construction

Stage 4 (`src/stage4_spiral_time/`) implements the full spiral-time embedding defined in manuscript §4. The spiral-time coordinate (Eq. 6):

```
Ψ(t) = t + i φ(t) + j χ(t)
```

is assembled from three components: ordinary forward time t, adaptive phase φ(t) (Eq. 3–4), and memory component χ(t) (Eq. 7).

**φ(t) computation** (`compute_phi.py`): Baseline-normalised absolute deviations Δk(t) = |raw_k − μ_k| / (σ_k + ε) are formed for each ICE channel. Adaptive weights w_k(t) = Δk / (ΔE + ΔI + ΔC + ε) are clipped to [0, 1] and renormalised, then:

```
φ(t) = 1 − (w_E · ΔE + w_I · ΔI + w_C · ΔC)
```

The weights Σ_k w_k(t) = 1 by construction (Eq. 4). The adaptive weighting *"dynamically emphasises the dominant instability component"* (manuscript §3).

**χ(t) computation** (`compute_chi.py`): The causal convolution

```
χ[n] = Δt · Σ_{m=0}^{n} K(t_n − t_m) · φ[m]
```

is evaluated as a lower-triangular matrix–vector product. The kernel matrix is row-wise normalised so that χ(t_n) is a proper weighted average of past φ values. Two kernels are supported:

- Exponential: K(s) = λ exp(−λs), s > 0 (default λ = 1.0)
- Gaussian: K(s) = (1/(σ√(2π))) exp(−s²/(2σ²)), s > 0 (default σ = 1.0)

The normalised χ(t) is standardised to zero mean and unit variance. The kernel-convolution structure is explicitly compatible with the generalised Langevin equation formalism (Eq. 8 of the manuscript) and the Mori–Zwanzig projection formalism (§4.1).

### Deterministic Simulation Layer

The synthetic ICE series used in Stage 4 is generated by `src/simulation/generate_synthetic.py` with default parameters: f_s = 50 Hz, T = 180 s, with four canonical regime boundaries:

| Regime | Time interval |
|---|---|
| Stable | [0, 45) s |
| Pre-instability | [45, 90) s |
| Instability | [90, 135) s |
| Recovery | [135, 180] s |

Each ICE component is constructed independently with distinct oscillation frequencies (E: 0.25 Hz, I: 0.30 Hz, C: 0.20 Hz) and regime-specific amplitude profiles. All components are z-score normalised against the stable baseline [0, 45) s. Outputs are saved to `data/synthetic/eic_timeseries.csv`, `data/synthetic/delta_phi.csv`, and `data/synthetic/regimes.csv`.

### Spiral-Time Trajectory Operator

The operator `D_Ψ f` is implemented in `spiral_operator.py` according to manuscript Eq. (12):

```
D_Ψ f(γ(t)) := d/dt f(γ(t))
              + i · φ̇(t) · v_φ(t) · ∇f(γ(t))
              + j · χ̇(t) · v_χ(t) · ∇f(γ(t))
```

Direction vectors v_φ = v_χ = γ̇(t) / |γ̇(t)| (normalised trajectory velocity). Spatial gradients ∇f are estimated via central finite differences (ε = 1×10⁻⁶). At stationary points (|γ̇| = 0), directional contributions are set to zero, consistent with the limiting case of manuscript Eq. (14): D_Ψ f → d/dt f(γ(t)).

Optional Savitzky–Golay pre-smoothing (window = 11, polyorder = 3) is applied to trajectory and phase signals before derivative computation to suppress noise-dominated estimates without altering the mathematical definition.

### Figure References

| Figure | Path | Content |
|---|---|---|
| φ(t) and χ(t) curves | `figures/stage4_spiral_time/phi_chi_curves.png` | Time series of adaptive phase φ(t) and memory component χ(t) over the synthetic ICE trajectory |
| Spiral embedding | `figures/stage4_spiral_time/spiral_embedding.png` | 3D or 2D projection of the spiral-time embedding Ψ(t), colour-coded by regime |

### Interpretation Constraints

Per manuscript §10, the following constraints apply to the interpretation of Stage 4 outputs:

- Ψ(t) is a *structured embedding coordinate* for dynamical analysis, not an ontological description of biological time.
- φ(t) and χ(t) should be understood as effective descriptors of delayed coupling, phase-history structure, and memory-like dynamical effects (Remark 1 of the manuscript).
- χ(t) is not a statistical autocorrelation; it represents a dynamical memory contribution in the effective state of the system.
- No universal threshold is implied; all thresholds must be determined empirically within the system under study.
- The Lyapunov-like functional V(t) = ½ ΔΦ(t)² (Eq. 15) provides a *local* stability interpretation only; no global Lyapunov function for arbitrary biological systems is asserted.

---

## 6. Stage 5 — Regime Segmentation

### Thresholds

Stage 5 (`src/stage5_regime_transitions/detect_regimes.py`) derives two empirical thresholds from the ΔΦ(t) series. Per manuscript §5:

> *"The threshold Φ_c is not assumed to be universal and must be determined empirically for each system under study."*

Default quantile levels:

| Threshold | Quantile | Separates |
|---|---|---|
| θ_low | q_low = 0.75 (75th percentile) | Stable from Pre-instability |
| θ_high | q_high = 0.90 (90th percentile) | Pre-instability from Instability |

Both thresholds are computed from the full supplied ΔΦ(t) series. Alternatively, `segmentation_utils.compute_threshold()` supports a `"baseline"` method: θ = μ_baseline + n_sigma · σ_baseline (default n_sigma = 2.0), consistent with the baseline normalisation in manuscript Eq. (27) and the stable-regime statistics protocol of §8.3.

### Hysteresis Rules

To suppress rapid label oscillations when ΔΦ(t) hovers near a regime boundary, offset exit thresholds are applied:

```
δ = (θ_high − θ_low) × hysteresis,  hysteresis = 0.05 (default)
θ_low_exit  = θ_low  − δ    (exit pre-instability → stable)
θ_high_exit = θ_high − δ    (exit instability → pre-instability)
```

Enter thresholds are the unmodified quantile values. Exit thresholds are lowered by 5% of the inter-threshold range.

### Minimum Dwell-Time

After hysteresis labelling and optional recovery detection, a minimum-dwell-time filter is applied:

- **min_dwell = 30 samples** (default)
- Any contiguous run of identical labels shorter than 30 samples is merged into the longer adjacent run.
- This ensures the segmentation output contains broad, stable regime bands rather than single-sample threshold crossings due to measurement noise.

The filter is implemented via iterative run-length encoding in `_enforce_min_dwell()`.

### Regime Definitions

The four canonical regimes follow manuscript §5 and the `RegimeLabels` model in `src/core/models.py`:

| Regime | Condition | Manuscript correspondence |
|---|---|---|
| **Stable** | ΔΦ(t) < θ_low | "proximity to the reference stability region" (§5); ΔΦ(t) < Φ_c |
| **Pre-instability** | θ_low ≤ ΔΦ(t) < θ_high | "increased sensitivity to perturbations" (§5); ΔΦ(t) ≈ Φ_c (metastable) |
| **Instability** | ΔΦ(t) ≥ θ_high | "sustained deviation from the reference regime" (§5); ΔΦ(t) > Φ_c |
| **Recovery** | ΔΦ(t) < θ_low *after* first instability epoch | "cellular recovery trajectories" (§6) |

Recovery detection is optional (`detect_recovery = True` by default). After the first instability time point, any subsequent point that would be labelled "stable" (ΔΦ < θ_low) is instead labelled "recovery", reflecting the system's return toward the reference stability region.

The three-regime structure (Stable / Pre-instability / Instability) without recovery detection is available by setting `detect_recovery = False`.

### Figure References

| Figure | Path | Content |
|---|---|---|
| ΔΦ thresholds | `figures/stage5_regime_transitions/delta_phi_thresholds.png` | ΔΦ(t) time series with θ_low and θ_high threshold lines |
| Regime segmentation | `figures/stage5_regime_transitions/regime_segmentation.png` | ΔΦ(t) with colour-coded regime shading (Stable / Pre-instability / Instability / Recovery) |

---

## 7. Table 3 Alignment

### Extracted Values from `simulations/table3_metrics.csv`

The file `simulations/table3_metrics.csv` contains quantitative performance metrics for four conditions:

| Condition | AUC_full | AUC_E | AUC_I | AUC_C | n_transitions | mean_stable | std_stable | mean_pre | mean_instability | mean_recovery |
|---|---|---|---|---|---|---|---|---|---|---|
| Observed (synthetic) | 0.9408 | 0.8498 | 0.8610 | 0.8412 | 3 | 1.6701 | 0.4592 | 3.1535 | 11.8932 | 7.0946 |
| Shuffled-time null | 0.4945 | — | — | — | 3 | 5.9242 | 4.5894 | 6.0617 | 5.9000 | 5.9253 |
| Phase-randomised null | 0.6366 | — | — | — | 3 | 6.9842 | 2.4657 | 6.1015 | 7.9811 | 7.5994 |
| Stable control | 0.4937 | — | — | — | 3 | 0.0789 | 0.0343 | 0.0799 | 0.0788 | 0.0795 |

### Verification Against Manuscript Table 3

Manuscript §8.8 Table 3 contains placeholder "XX" values for all metrics; no numerical entries are present in the current manuscript draft. The computed values from `simulations/table3_metrics.csv` are therefore the primary source of quantitative evidence for Table 3.

**Key observations:**

1. **AUC_full (Observed):** 0.9408 — the triadic ΔΦ(t) achieves strong discriminative performance between instability and non-instability regimes.
2. **Component AUCs:** AUC_E = 0.8498, AUC_I = 0.8610, AUC_C = 0.8412 — all below AUC_full, confirming that the triadic combination provides discriminative advantage over single-component indicators (manuscript §8.1).
3. **Shuffled-time null AUC = 0.4945** — near chance level (0.50), confirming that temporal ordering is essential for ΔΦ(t) to exhibit regime separation (manuscript §9.2: "temporal structure is removed, leading to a collapse of ordered regime transitions").
4. **Phase-randomised null AUC = 0.6366** — above chance but substantially below the observed value, consistent with manuscript §9.2: "transition signatures are significantly attenuated".
5. **Stable control AUC = 0.4937** — near chance, confirming no spurious detection in a system without true transition structure (manuscript §9.1).
6. **Stable control mean ΔΦ values** remain uniformly low across all regime segments (0.079–0.080), confirming ΔΦ does not exhibit systematic increase in the absence of genuine transitions.

**Multi-seed benchmark comparison** (from `data/stage3/stage_iii_summary.csv`): The Stage 3 benchmark over 50 seeds yields AUC_full = 0.9998 ± 0.0002. The discrepancy from the simulations/table3_metrics.csv value of 0.9408 reflects a difference in signal generation: Stage 3 uses the `simulate_signal()` function with a seed-varying random process and a 90th-percentile threshold per seed; `table3_metrics.csv` uses the `generate_synthetic_ice()` deterministic four-regime signal. Both datasets are internally consistent with the validation protocol described in §8.3 and §9.

### Discrepancy Note

The manuscript Table 3 currently contains only "XX" placeholders. The computed values documented above constitute the candidate final values for publication. The Stage 3 multi-seed results (AUC_full ≈ 0.9998) and the simulation-level results (AUC_full ≈ 0.94) arise from different synthetic protocols and are not contradictory; they reflect different signal complexity levels. The manuscript Table 3 label ("respiratory synthetic system") aligns most directly with the `generate_synthetic_ice()` pipeline values in `simulations/table3_metrics.csv`.

---

## 8. Reproducibility Notes

### How to Run the Pipeline End-to-End

**Dependencies** (`requirements.txt`):

```
numpy
scipy
matplotlib
pandas
scikit-learn
```

Install with: `pip install -r requirements.txt`

**Execution order:**

```bash
# Stage 1 — preprocessing, feature extraction, ΔΦ computation
python src/stage1/stage_i.py

# Stage 2 — smoothing and regime segmentation
python src/stage2/stage_ii.py

# Stage 3 — multi-seed ground-truth validation
python src/stage3/stage_iii.py

# Synthetic data and null tests (prerequisite for Stage 4 and simulations)
python src/simulation/generate_synthetic.py
python src/simulation/null_tests.py

# Stage 4 — spiral-time embedding figures
python src/stage4_spiral_time/generate_figures.py

# Stage 5 — regime-transition figures
python src/stage5_regime_transitions/generate_figures.py
```

### Expected Outputs

| Stage | Expected files |
|---|---|
| Stage 1 | `data/stage1/stage_i_results.csv` (8 601 rows), `figures/stage1/stage_i_figure.png` |
| Stage 2 | `data/stage2/stage_ii_segmentation.csv` (8 601 rows), `figures/stage2/stage_ii_regimes.png` |
| Stage 3 | `data/stage3/stage_iii_summary.csv` (51 rows: 50 seeds + mean±std), `figures/stage3/stage_iii_seed0.png` |
| Simulation | `data/synthetic/eic_timeseries.csv`, `data/synthetic/delta_phi.csv`, `data/synthetic/regimes.csv`, `data/synthetic/null_tests/{shuffled,surrogate,stable_control}.csv` |
| Stage 4 | `figures/stage4_spiral_time/phi_chi_curves.png`, `figures/stage4_spiral_time/spiral_embedding.png` |
| Stage 5 | `figures/stage5_regime_transitions/delta_phi_thresholds.png`, `figures/stage5_regime_transitions/regime_segmentation.png` |

All random processes use fixed seeds (`numpy.random.default_rng(seed=42)` in Stage 1; per-seed `rng = np.random.default_rng(seed)` in Stage 3; `seed=0` in simulation modules) to ensure full bit-for-bit reproducibility across runs.

### Versioning and Data Availability

**Synthetic data only.** Per manuscript Data Availability statement:

> *"No experimental datasets were generated in this study. The present manuscript is theoretical and methodological in scope. Synthetic simulations and future in vitro datasets are intended for subsequent work."*

All data files in the repository (`data/`, `simulations/`) are generated programmatically by the pipeline scripts and can be regenerated from source code without external data dependencies. The repository constitutes the complete and self-contained computational record for Stages 1–5.

Extension modules for Stages 6 (cellular) and 7 (DNA-associated dynamics) are available under `src/extensions/` but are outside the scope of this validation document.

---

*Document generated from repository sources only. No speculative or forward-looking content regarding Stages 6–7 is included. All terminology follows PAPER.md exactly.*
