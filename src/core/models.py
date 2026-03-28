"""
Core data models for the triadic stability framework.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ICEStateSeries:
    """Time-resolved ICE (Energy, Information, Coherence) deviation series.

    Each component holds the normalized deviation from the reference state
    as defined in the manuscript:

        ΔE(t), ΔI(t), ΔC(t)

    After z-score normalization against a stable baseline the reference state
    corresponds to x* = (0, 0, 0), so these arrays are identical to the
    normalized feature values produced by the upstream pipeline stages.

    Attributes
    ----------
    delta_E : np.ndarray, shape (T,)
        Normalized energy deviation ΔE(t).
    delta_I : np.ndarray, shape (T,)
        Normalized information deviation ΔI(t).
    delta_C : np.ndarray, shape (T,)
        Normalized coherence deviation ΔC(t).
    """

    delta_E: np.ndarray
    delta_I: np.ndarray
    delta_C: np.ndarray

    def __post_init__(self) -> None:
        self.delta_E = np.asarray(self.delta_E, dtype=float)
        self.delta_I = np.asarray(self.delta_I, dtype=float)
        self.delta_C = np.asarray(self.delta_C, dtype=float)
        if not (self.delta_E.shape == self.delta_I.shape == self.delta_C.shape):
            raise ValueError(
                "delta_E, delta_I, and delta_C must have the same shape; "
                f"got {self.delta_E.shape}, {self.delta_I.shape}, {self.delta_C.shape}."
            )


# Valid regime label strings (Section 5 of the manuscript).
_REGIME_LABELS = frozenset({"stable", "pre-instability", "instability", "recovery"})


@dataclass
class RegimeLabels:
    """Per-time-point regime assignments derived from the instability magnitude.

    The four possible label values follow the regime taxonomy defined in
    Section 5 of the manuscript:

    - ``"stable"``          – ΔΦ(t) < θ_low; proximity to the reference
                              stability region.
    - ``"pre-instability"`` – θ_low ≤ ΔΦ(t) < θ_high; elevated sensitivity
                              to perturbations (metastable in manuscript
                              terminology).
    - ``"instability"``     – ΔΦ(t) ≥ θ_high; sustained deviation from the
                              reference regime.
    - ``"recovery"``        – (optional) ΔΦ(t) < θ_low *after* the first
                              instability epoch; the system is returning toward
                              the stable reference region.

    Attributes
    ----------
    labels : np.ndarray, shape (T,), dtype object
        Per-time-point regime strings drawn from the set
        ``{"stable", "pre-instability", "instability", "recovery"}``.
    theta_low : float
        Lower quantile threshold separating *stable* from *pre-instability*.
        Derived empirically from the ΔΦ(t) series (manuscript Section 5).
    theta_high : float
        Upper quantile threshold separating *pre-instability* from
        *instability*.  Derived empirically from the ΔΦ(t) series.
    """

    labels: np.ndarray
    theta_low: float
    theta_high: float

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels, dtype=object)
        if self.labels.ndim != 1:
            raise ValueError(
                f"labels must be a 1-D array; got shape {self.labels.shape}."
            )
        invalid = set(np.unique(self.labels)) - _REGIME_LABELS
        if invalid:
            raise ValueError(
                f"labels contains invalid regime strings: {invalid}. "
                f"Allowed values are {sorted(_REGIME_LABELS)}."
            )
        if not (0.0 <= self.theta_low <= self.theta_high):
            raise ValueError(
                f"Thresholds must satisfy 0 ≤ theta_low ≤ theta_high; "
                f"got theta_low={self.theta_low}, theta_high={self.theta_high}."
            )
