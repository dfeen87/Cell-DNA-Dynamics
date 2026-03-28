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

Numerical stability
-------------------
When the sum ΔE + ΔI + ΔC is near zero the adaptive weights are
ill-defined (all deviations cancel).  At those time points φ(t) is set
to 1.0, consistent with zero net deviation from the reference state.
The threshold `eps` controls how small "near zero" means; the default
is conservative enough for double-precision arrays of any length.
"""

from __future__ import annotations

import sys
import os

import numpy as np

# Allow running from the repository root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.core.models import ICEStateSeries


def compute_phi(series: ICEStateSeries, eps: float = 1e-12) -> np.ndarray:
    """Compute the adaptive phase φ(t) (Eq. 3–4 of the manuscript).

    Parameters
    ----------
    series : ICEStateSeries
        ICE deviation series containing ΔE(t), ΔI(t), ΔC(t).
    eps : float, optional
        Threshold below which the denominator |ΔE + ΔI + ΔC| is
        considered numerically zero.  Defaults to 1e-12.

    Returns
    -------
    phi : np.ndarray, shape (T,)
        Adaptive phase values φ(t).  Equal to 1.0 at time points where
        the denominator is below *eps* (reference-state proximity).
    """
    dE = series.delta_E
    dI = series.delta_I
    dC = series.delta_C

    # Numerator:   ΔE² + ΔI² + ΔC²   (squared L2 norm of deviation vector)
    numerator = dE * dE + dI * dI + dC * dC

    # Denominator: ΔE + ΔI + ΔC       (L1-style signed sum, per Eq. 4)
    denominator = dE + dI + dC

    # Stable division: fall back to 1.0 where |denominator| < eps.
    # The boolean mask is computed once and reused to avoid redundant
    # evaluation, which matters for very long time series.
    valid = np.abs(denominator) >= eps
    safe_denom = np.where(valid, denominator, 1.0)
    phi = np.where(valid, 1.0 - numerator / safe_denom, 1.0)

    return phi
