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

Numerical stability (Stage 4 cleanup)
--------------------------------------
The kernel is row-wise normalised so that each χ(t_n) is a proper weighted
average of past φ values rather than an unnormalised sum.  After the
convolution, χ(t) is standardised (zero mean, unit variance) so that the
axis is on a stable scale for derivative computation and plotting.

The raw (pre-standardisation) χ(t) is optionally saved to a debug directory.
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
    standardize: bool = True,
    debug_dir: str | None = None,
    **kernel_kwargs: float,
) -> np.ndarray:
    """Compute the memory component χ(t) (Eq. 7 of the manuscript).

    χ(t) = ∫₀ᵗ K(t − τ) φ(τ) dτ

    The integral is evaluated as a discrete causal convolution over the
    supplied time grid *t*.  Strict causality is guaranteed: for each
    output index n only values φ[m] with t[m] ≤ t[n] are included.

    The kernel is row-wise normalised so that χ(t_n) is a proper weighted
    average of past φ values (the kernel weight sum equals 1 at each n).
    After the convolution, χ(t) is standardised to zero mean and unit
    variance when *standardize* is ``True`` (default), providing a stable
    scale for derivative computation and plotting.

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
    standardize : bool, optional
        If ``True`` (default), standardise χ(t) to zero mean and unit
        variance after the convolution.  The raw values are saved to
        *debug_dir* when provided.  Set to ``False`` only if you need
        the raw convolution output.
    debug_dir : str or None, optional
        If provided, the raw (pre-standardisation) χ(t) array is saved as
        ``chi_raw.npy`` in this directory.  The directory is created if it
        does not exist.  Defaults to ``None`` (no saving).
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
        Standardised (zero mean, unit std) when *standardize* is ``True``.

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

    # ── Row-wise kernel normalisation ─────────────────────────────────────────
    # Normalise each row so that Σ_m K_norm[n, m] * dt[m] = 1, making χ(t_n)
    # a proper weighted average of past φ values.  This keeps χ bounded by
    # the range of φ regardless of the kernel width or time-grid spacing.
    row_norms = K_matrix @ dt          # shape (T,); integral of K up to t[n]
    safe_norms = np.where(row_norms > 1e-12, row_norms, 1.0)
    K_normalised = K_matrix / safe_norms[:, np.newaxis]

    # χ[n] = Σ_m K_normalised[n, m] · φ[m] · dt[m]
    chi = K_normalised @ (phi * dt)    # shape (T,)

    # ── Optional debug save (raw, before standardisation) ─────────────────────
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        np.save(os.path.join(debug_dir, "chi_raw.npy"), chi)

    # ── Standardise χ(t) to zero mean, unit variance ──────────────────────────
    if standardize:
        chi_std = np.std(chi)
        chi = (chi - np.mean(chi)) / (chi_std + 1e-12)

    return chi
