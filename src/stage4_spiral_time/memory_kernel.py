"""
Memory kernels K(s) for the spiral-time embedding – Stage 4
============================================================

Provides the strictly causal memory kernels referenced in the manuscript
(Eq. 7).  All kernels are defined on the open interval s ∈ (0, ∞) and are
identically zero for s ≤ 0 (strict causality: only strictly past values
τ < t contribute to χ(t)).

Supported kernels
-----------------
exponential_kernel(s, lam)
    K(s) = λ · exp(−λ s),  s > 0; else 0.
    Equivalent to a first-order memory decay with rate λ > 0.

gaussian_kernel(s, sigma)
    K(s) = (1 / (σ √(2π))) · exp(−s² / (2σ²)),  s > 0; else 0.
    One-sided Gaussian weighting truncated to the causal half-line.

Both functions accept array-like *s* and return a numpy array of the same
shape so they can be used directly inside numerical integration loops or
vectorised convolutions.
"""

from __future__ import annotations

import numpy as np


def exponential_kernel(s: "np.ndarray | float", lam: float) -> np.ndarray:
    """Strictly causal exponential memory kernel.

    Parameters
    ----------
    s : array-like or float
        Time lag values s = t − τ.  Must be strictly positive for a causal
        evaluation (values s ≤ 0 return 0).
    lam : float
        Decay rate λ > 0.  Larger values correspond to shorter memory.

    Returns
    -------
    K : np.ndarray
        Kernel evaluated at each element of *s*.  Shape matches input.

    Raises
    ------
    ValueError
        If *lam* is not strictly positive.
    """
    if lam <= 0:
        raise ValueError(f"Decay rate lam must be positive; got {lam!r}.")

    s = np.asarray(s, dtype=float)
    K = np.where(s > 0, lam * np.exp(-lam * s), 0.0)
    return K


def gaussian_kernel(s: "np.ndarray | float", sigma: float) -> np.ndarray:
    """Strictly causal (one-sided) Gaussian memory kernel.

    Parameters
    ----------
    s : array-like or float
        Time lag values s = t − τ.  Values s ≤ 0 return 0.
    sigma : float
        Standard deviation σ > 0 of the Gaussian envelope.

    Returns
    -------
    K : np.ndarray
        Kernel evaluated at each element of *s*.  Shape matches input.

    Raises
    ------
    ValueError
        If *sigma* is not strictly positive.
    """
    if sigma <= 0:
        raise ValueError(f"Standard deviation sigma must be positive; got {sigma!r}.")

    s = np.asarray(s, dtype=float)
    # Compute in log-space to avoid inf × 0 = nan when sigma is very small:
    #   log K(s) = -0.5 * log(2π) - log(σ) - s² / (2σ²)
    log_norm = -0.5 * np.log(2.0 * np.pi) - np.log(sigma)
    log_exp = -(s**2) / (2.0 * sigma**2)
    K = np.where(s > 0, np.exp(log_norm + log_exp), 0.0)
    return K
