"""
Numerical utility functions for the triadic stability framework.
"""

from __future__ import annotations

import numpy as np


def numerical_derivative(x: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
    """Compute the numerical time derivative of a 1-D array.

    Uses ``numpy.gradient``, which applies second-order central differences
    at interior points and first-order (forward/backward) differences at
    the boundaries.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        The signal to differentiate.
    t : np.ndarray, shape (T,), optional
        Monotonically increasing time coordinates corresponding to *x*.
        When provided the derivative is scaled by the actual time spacing
        (non-uniform grids are supported).  When omitted, unit spacing is
        assumed (i.e. the derivative is with respect to the sample index).

    Returns
    -------
    dx_dt : np.ndarray, shape (T,)
        Numerical derivative of *x* with the same shape as the input.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 2 * np.pi, 500)
    >>> x = np.sin(t)
    >>> dx_dt = numerical_derivative(x, t)
    >>> np.allclose(dx_dt, np.cos(t), atol=1e-3)
    True
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(
            f"x must be a 1-D array; got shape {x.shape}."
        )
    if t is None:
        return np.gradient(x)
    t = np.asarray(t, dtype=float)
    if t.shape != x.shape:
        raise ValueError(
            f"t and x must have the same shape; got {t.shape} and {x.shape}."
        )
    return np.gradient(x, t)
