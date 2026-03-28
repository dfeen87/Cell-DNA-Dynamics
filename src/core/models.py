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
