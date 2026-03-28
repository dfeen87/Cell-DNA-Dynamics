"""
Memory component χ(t) – Stage 4 Spiral-Time Embedding
======================================================

Implements the memory component χ(t) as defined in the manuscript (Eq. 7):

    χ(t) = ∫₀ᵗ K(t − τ) φ(τ) dτ                               (7)

where K(t − τ) is a strictly causal memory kernel (K(s) = 0 for s < 0)
and φ(τ) is the instantaneous phase component computed by compute_phi.

Only manuscript-consistent kernels are supported (exponential, Gaussian);
both are imported from memory_kernel.py in the same package.

Numerical method
----------------
The integral is evaluated on a uniform time grid with spacing Δt using a
discrete causal convolution:

    χ[n] = Δt · Σ_{m=0}^{n} K(t_n − t_m) · φ[m]

This is equivalent to the full causal integral when the kernel is evaluated
at the same discrete lags.  The sum runs from m = 0 to m = n so that only
past (and present) values contribute, which preserves strict causality.

For long time series the inner summation is implemented as a vectorised
matrix–vector product, keeping memory usage at O(T²) for a length-T series.
"""

from __future__ import annotations

import sys
import os
from typing import Callable, Literal

import numpy as np

# Allow running from the repository root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.stage4_spiral_time.memory_kernel import exponential_kernel, gaussian_kernel

# Mapping from kernel name to callable for public API convenience.
_KERNEL_REGISTRY: dict[str, Callable[..., np.ndarray]] = {
    "exponential": exponential_kernel,
    "gaussian": gaussian_kernel,
}

KernelName = Literal["exponential", "gaussian"]


def compute_chi(
    phi: np.ndarray,
    t: np.ndarray,
    kernel: KernelName | Callable[..., np.ndarray] = "exponential",
    **kernel_kwargs: float,
) -> np.ndarray:
    """Compute the memory component χ(t) (Eq. 7 of the manuscript).

    χ(t) = ∫₀ᵗ K(t − τ) φ(τ) dτ

    The integral is evaluated as a discrete causal convolution over the
    supplied time grid *t*.  Strict causality is guaranteed: for each
    output index n only values φ[m] with t[m] ≤ t[n] are included.

    Parameters
    ----------
    phi : np.ndarray, shape (T,)
        Instantaneous phase component φ(t) computed by compute_phi.
    t : np.ndarray, shape (T,)
        Monotonically increasing time coordinates.  Must satisfy
        ``t.shape == phi.shape``.
    kernel : {"exponential", "gaussian"} or callable, optional
        Memory kernel to use.  Can be either one of the built-in name
        strings or any callable with signature ``K(s, **kernel_kwargs)``
        that accepts an array of non-negative lag values and returns an
        array of the same shape.  Defaults to ``"exponential"``.
    **kernel_kwargs
        Additional keyword arguments forwarded to the kernel function.
        For the built-in kernels:

        - ``"exponential"``: requires ``lam`` (decay rate λ > 0).
        - ``"gaussian"``:    requires ``sigma`` (standard deviation σ > 0).

        If no ``kernel_kwargs`` are provided the defaults are
        ``lam=1.0`` for the exponential kernel and ``sigma=1.0`` for
        the Gaussian kernel.

    Returns
    -------
    chi : np.ndarray, shape (T,)
        Memory component χ(t) evaluated at each time point.

    Raises
    ------
    ValueError
        If *phi* and *t* have different shapes, or if *t* is not
        one-dimensional.
    TypeError
        If *kernel* is neither a recognised name string nor a callable.

    Examples
    --------
    >>> import numpy as np
    >>> from src.stage4_spiral_time.compute_chi import compute_chi
    >>> t = np.linspace(0, 10, 200)
    >>> phi = np.sin(t)
    >>> chi = compute_chi(phi, t, kernel="exponential", lam=0.5)
    >>> chi.shape
    (200,)
    """
    phi = np.asarray(phi, dtype=float)
    t = np.asarray(t, dtype=float)

    if phi.ndim != 1:
        raise ValueError(
            f"phi must be a 1-D array; got shape {phi.shape}."
        )
    if t.ndim != 1:
        raise ValueError(
            f"t must be a 1-D array; got shape {t.shape}."
        )
    if phi.shape != t.shape:
        raise ValueError(
            f"phi and t must have the same shape; got {phi.shape} and {t.shape}."
        )

    # Resolve the kernel callable.
    if isinstance(kernel, str):
        if kernel not in _KERNEL_REGISTRY:
            raise ValueError(
                f"Unknown kernel {kernel!r}.  "
                f"Choose from {sorted(_KERNEL_REGISTRY)!r} or pass a callable."
            )
        kernel_fn = _KERNEL_REGISTRY[kernel]
        # Supply sensible defaults when the caller omits kernel parameters.
        if not kernel_kwargs:
            if kernel == "exponential":
                kernel_kwargs = {"lam": 1.0}
            elif kernel == "gaussian":
                kernel_kwargs = {"sigma": 1.0}
    elif callable(kernel):
        kernel_fn = kernel
    else:
        raise TypeError(
            f"kernel must be a string or callable; got {type(kernel)!r}."
        )

    T = len(t)

    # Build the (T × T) lower-triangular lag matrix:
    #   lags[n, m] = t[n] − t[m]  for m ≤ n,  else 0
    # The strictly causal mask sets lags[n, m] = 0 for m > n, which
    # evaluates K(0) = K_max there – but those entries are multiplied out
    # by the causal mask below so they never contribute to the sum.
    t_col = t[:, np.newaxis]   # shape (T, 1)
    t_row = t[np.newaxis, :]   # shape (1, T)
    lags = t_col - t_row       # shape (T, T);  lags[n, m] = t[n] - t[m]

    # Strict-causality mask: m ≤ n  ↔  t[m] ≤ t[n]  ↔  lags[n, m] ≥ 0.
    causal_mask = lags >= 0.0  # shape (T, T), True in lower triangle + diag

    # Evaluate K only on non-negative lags; elsewhere the mask zeroes it.
    K_matrix = np.where(causal_mask, kernel_fn(lags, **kernel_kwargs), 0.0)

    # Numerical integration weight: use trapezoidal weights along τ-axis.
    # For a non-uniform grid dt[m] = t[m+1] - t[m-1]) / 2 (centred),
    # with half-interval corrections at the boundaries.
    dt = np.empty(T)
    if T == 1:
        dt[0] = 1.0
    else:
        dt[0] = t[1] - t[0]
        dt[-1] = t[-1] - t[-2]
        dt[1:-1] = (t[2:] - t[:-2]) / 2.0

    # χ[n] = Σ_m K_matrix[n, m] · φ[m] · dt[m]
    chi = K_matrix @ (phi * dt)  # shape (T,)

    return chi
