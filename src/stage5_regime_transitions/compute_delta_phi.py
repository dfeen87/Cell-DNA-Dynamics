"""
Instability magnitude ΔΦ(t) – Stage 5 Regime-Transition Detection
==================================================================

Implements the instability magnitude defined in the manuscript (Eq. 2):

    ΔΦ(t) = sqrt( ΔE(t)² + ΔI(t)² + ΔC(t)² )      (2)

This quantity measures the Euclidean distance of the current ICE state from
the reference state x* = (0, 0, 0) in the reduced ICE manifold.  After
z-score normalization the reference state corresponds to the stable baseline,
so ΔΦ(t) is a parameter-free, instantaneous indicator of instability.

The numerical time-derivative utility :func:`~src.core.utils.numerical_derivative`
from :mod:`src.core.utils` is available for downstream computations such as
d/dt ΔΦ(t) and the Lyapunov functional derivative dV/dt = ΔΦ(t) · d/dt ΔΦ(t)
(manuscript Eq. 16).

References
----------
Manuscript Eq. (2):  ΔΦ(t) = sqrt( ΔE(t)² + ΔI(t)² + ΔC(t)² )
"""

from __future__ import annotations

import sys
import os

import numpy as np

# Allow running from the repository root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.core.models import ICEStateSeries
from src.core.utils import numerical_derivative  # noqa: F401 – re-exported for callers


def compute_delta_phi(series: ICEStateSeries) -> np.ndarray:
    """Compute the instability magnitude ΔΦ(t) (Eq. 2 of the manuscript).

    The instability magnitude is the Euclidean norm of the ICE deviation
    vector at each time step:

        ΔΦ(t) = sqrt( ΔE(t)² + ΔI(t)² + ΔC(t)² )

    Parameters
    ----------
    series : ICEStateSeries
        Time-resolved ICE deviation series.  Each component
        (``delta_E``, ``delta_I``, ``delta_C``) holds the normalized
        deviation ΔE(t), ΔI(t), ΔC(t) from the reference state as defined
        in the manuscript.

    Returns
    -------
    delta_phi : np.ndarray, shape (T,)
        Instability magnitude ΔΦ(t) ≥ 0 at each time step.

    Notes
    -----
    To compute the time-derivative of ΔΦ(t) (e.g. for the Lyapunov
    functional dV/dt, manuscript Eq. 16), use the
    :func:`~src.core.utils.numerical_derivative` utility that is
    re-exported from this module::

        from src.stage5_regime_transitions.compute_delta_phi import (
            compute_delta_phi,
            numerical_derivative,
        )
        delta_phi = compute_delta_phi(series)
        d_delta_phi_dt = numerical_derivative(delta_phi, t)

    Examples
    --------
    >>> import numpy as np
    >>> from src.core.models import ICEStateSeries
    >>> from src.stage5_regime_transitions.compute_delta_phi import compute_delta_phi
    >>> series = ICEStateSeries(
    ...     delta_E=np.array([0.0, 1.0, 0.0]),
    ...     delta_I=np.array([0.0, 0.0, 1.0]),
    ...     delta_C=np.array([0.0, 0.0, 0.0]),
    ... )
    >>> compute_delta_phi(series)
    array([0., 1., 1.])
    """
    dE = series.delta_E
    dI = series.delta_I
    dC = series.delta_C

    # ΔΦ(t) = sqrt( ΔE(t)² + ΔI(t)² + ΔC(t)² )   (Eq. 2)
    return np.sqrt(dE ** 2 + dI ** 2 + dC ** 2)
