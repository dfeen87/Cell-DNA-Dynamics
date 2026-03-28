"""
Adaptive phase φ(t) – Stage 4 Spiral-Time Embedding
=====================================================

Implements the adaptive phase definition from the manuscript (Eq. 3–4):

    φ(t) = 1 − Σ_{k ∈ {E,I,C}} w_k(t) Δk(t)               (3)

    w_k(t) = Δk(t) / ( ΔE(t) + ΔI(t) + ΔC(t) )             (4)

Substituting (4) into (3):

    φ(t) = 1 − ( ΔE(t)² + ΔI(t)² + ΔC(t)² )
               ─────────────────────────────────
               ( ΔE(t)  + ΔI(t)  + ΔC(t) )

The weights satisfy Σ_k w_k(t) = 1 by construction wherever the
denominator is non-zero.

Numerical stability (Stage 4 cleanup)
--------------------------------------
Deviations Δk(t) are computed as baseline-normalised absolute deviations
so that all three channels are on a comparable, non-negative scale before
the weights are formed.  The denominator is guarded by ε = 1e-8 and the
resulting weights are clipped to [0, 1] and renormalised, eliminating
numerical blow-up even when the three channels move in the same direction.

The unclipped φ(t) is optionally saved to a caller-supplied debug
directory.  A standalone helper :func:`robust_clip_for_plot` applies
1st–99th percentile clipping for manuscript-ready figures.
"""

from __future__ import annotations

import sys
import os

import numpy as np

# Allow running from the repository root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.core.models import ICEStateSeries


def robust_clip_for_plot(arr: np.ndarray, plo: float = 1.0, phi: float = 99.0) -> np.ndarray:
    """Clip *arr* to the [*plo*, *phi*] percentile range for plotting.

    Values outside the range are clipped (not removed), so the output
    array has the same length as the input.

    Parameters
    ----------
    arr : np.ndarray
        1-D array to clip.
    plo, phi : float
        Lower and upper percentile bounds (default 1st and 99th).

    Returns
    -------
    clipped : np.ndarray
        Array with the same shape as *arr*, clipped to [P_lo, P_hi].
    """
    lo = np.percentile(arr, plo)
    hi = np.percentile(arr, phi)
    return np.clip(arr, lo, hi)


def compute_phi(
    series: ICEStateSeries,
    eps: float = 1e-8,
    baseline_frac: float = 0.1,
    debug_dir: str | None = None,
) -> np.ndarray:
    """Compute the adaptive phase φ(t) (Eq. 3–4 of the manuscript).

    The mathematical definition is unchanged from the manuscript.  This
    function stabilises the numerics by:

    1. Forming baseline-normalised absolute deviations Δk(t) so that all
       three ICE channels are on a comparable, non-negative scale.
    2. Computing adaptive weights with a small ε = 1e-8 in the denominator,
       clipped to [0, 1] and renormalised to sum to 1.

    Parameters
    ----------
    series : ICEStateSeries
        ICE deviation series containing ΔE(t), ΔI(t), ΔC(t).
    eps : float, optional
        Small constant added to denominators to prevent division by zero.
        Defaults to 1e-8.
    baseline_frac : float, optional
        Fraction of the time series used as the baseline window to compute
        per-channel mean and standard deviation.  Defaults to 0.1 (first
        10 % of the series).
    debug_dir : str or None, optional
        If provided, the unclipped φ(t) array is saved as a ``.npy`` file
        in this directory under the name ``phi_raw.npy``.  The directory is
        created if it does not exist.  Defaults to ``None`` (no saving).

    Returns
    -------
    phi : np.ndarray, shape (T,)
        Adaptive phase values φ(t), unclipped.  Use
        :func:`robust_clip_for_plot` before plotting.
    """
    raw_E = series.delta_E
    raw_I = series.delta_I
    raw_C = series.delta_C

    T = len(raw_E)
    baseline_len = max(2, int(T * baseline_frac))

    # ── Baseline statistics (mean and std over the initial window) ─────────────
    mu_E = np.mean(raw_E[:baseline_len])
    mu_I = np.mean(raw_I[:baseline_len])
    mu_C = np.mean(raw_C[:baseline_len])

    sig_E = np.std(raw_E[:baseline_len])
    sig_I = np.std(raw_I[:baseline_len])
    sig_C = np.std(raw_C[:baseline_len])

    # ── Baseline-normalised absolute deviations Δk(t) ──────────────────────────
    # Dividing by (σ_k + ε) makes channels comparable and ensures Δk ≥ 0,
    # which guarantees all weights remain in [0, 1].
    dE = np.abs(raw_E - mu_E) / (sig_E + eps)
    dI = np.abs(raw_I - mu_I) / (sig_I + eps)
    dC = np.abs(raw_C - mu_C) / (sig_C + eps)

    # ── Adaptive weights w_k(t) = Δk / (ΔE + ΔI + ΔC + ε) ───────────────────
    denom = dE + dI + dC + eps

    w_E = dE / denom
    w_I = dI / denom
    w_C = dC / denom

    # Clip to [0, 1] (guarantees valid probabilities after renormalisation).
    w_E = np.clip(w_E, 0.0, 1.0)
    w_I = np.clip(w_I, 0.0, 1.0)
    w_C = np.clip(w_C, 0.0, 1.0)

    # Renormalise so that Σ_k w_k = 1 at every time point.
    w_sum = w_E + w_I + w_C + eps
    w_E = w_E / w_sum
    w_I = w_I / w_sum
    w_C = w_C / w_sum

    # ── φ(t) = 1 − Σ_k w_k(t) Δk(t)  (Eq. 3) ─────────────────────────────────
    phi = 1.0 - (w_E * dE + w_I * dI + w_C * dC)

    # ── Optional debug save ────────────────────────────────────────────────────
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        np.save(os.path.join(debug_dir, "phi_raw.npy"), phi)

    return phi
