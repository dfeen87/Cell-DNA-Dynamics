"""
Regime segmentation from ΔΦ(t) – Stage 5 Regime-Transition Detection
======================================================================

Implements the four-regime labelling scheme defined in Section 5 of the
manuscript:

    Stable regime:          ΔΦ(t) < θ_low
    Pre-instability regime: θ_low ≤ ΔΦ(t) < θ_high
    Instability regime:     ΔΦ(t) ≥ θ_high
    Recovery (optional):    ΔΦ(t) < θ_low after the first instability epoch

The two thresholds (θ_low, θ_high) are derived from the empirical quantiles
of the input ΔΦ(t) series, consistent with the quantile-based empirical
characterisation described in Section 5 and Section 6 of the manuscript:

    "The threshold Φ_c is not assumed to be universal and must be determined
     empirically for each system under study, for example through baseline
     characterisation, surrogate analysis, or controlled perturbation
     experiments."  (manuscript Section 5)

    "Alternative normalisation schemes (e.g. min–max scaling or
     quantile-based normalisation) may also be used, provided they are
     applied consistently." (manuscript Section 6)

No universal biological constants are used; every threshold is derived
from the supplied ΔΦ(t) series.

References
----------
Manuscript Section 5 (Regime Transition Structure):

    Stable:          ΔΦ(t) < Φ_c
    Metastable:      ΔΦ(t) ≈ Φ_c   → labelled "pre-instability"
    Unstable:        ΔΦ(t) > Φ_c   → labelled "instability"
"""

from __future__ import annotations

import sys
import os

import numpy as np

# Allow running from the repository root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.core.models import RegimeLabels


def detect_regimes(
    delta_phi: np.ndarray,
    q_low: float = 0.75,
    q_high: float = 0.90,
    detect_recovery: bool = True,
) -> RegimeLabels:
    """Segment ΔΦ(t) into regime labels following the manuscript's rules.

    Thresholds are derived from the empirical quantiles of the ΔΦ(t) series
    as the manuscript's quantile-based empirical approach (Section 5 and 6).
    No universal biological constants are assumed.

    Regime logic (Section 5 of the manuscript)
    ------------------------------------------
    Let θ_low = quantile(ΔΦ, q_low) and θ_high = quantile(ΔΦ, q_high).

    At each time step *t*:

    - ``"stable"``          if ΔΦ(t) < θ_low
    - ``"pre-instability"`` if θ_low ≤ ΔΦ(t) < θ_high
    - ``"instability"``     if ΔΦ(t) ≥ θ_high

    Optional recovery detection (when *detect_recovery* is ``True``):
    after the first time point at which ``"instability"`` is assigned,
    any subsequent time point whose raw label would be ``"stable"``
    (i.e. ΔΦ(t) < θ_low) is instead labelled ``"recovery"``, reflecting
    the system's return toward the reference stability region (Section 6:
    "cellular recovery trajectories").

    Parameters
    ----------
    delta_phi : array_like, shape (T,)
        Instability magnitude ΔΦ(t) ≥ 0 as produced by
        :func:`~src.stage5_regime_transitions.compute_delta_phi.compute_delta_phi`.
    q_low : float, default 0.75
        Quantile used for the lower threshold θ_low.  Must be in [0, 1).
        Separates *stable* from *pre-instability*.
    q_high : float, default 0.90
        Quantile used for the upper threshold θ_high.  Must be in (0, 1]
        and satisfy q_high > q_low.  Separates *pre-instability* from
        *instability*.
    detect_recovery : bool, default True
        Whether to apply the optional recovery-labelling post-processing
        step.  Set to ``False`` to obtain the three-regime version
        (stable / pre-instability / instability only).

    Returns
    -------
    RegimeLabels
        Per-time-point regime strings together with the derived thresholds
        θ_low and θ_high.

    Raises
    ------
    ValueError
        If *delta_phi* is not 1-D, or if the quantile parameters are
        outside admissible bounds or violate q_low < q_high.

    Examples
    --------
    >>> import numpy as np
    >>> from src.stage5_regime_transitions.detect_regimes import detect_regimes
    >>> rng = np.random.default_rng(0)
    >>> delta_phi = np.concatenate([
    ...     rng.normal(0.5, 0.1, 60),   # stable
    ...     rng.normal(1.5, 0.2, 60),   # pre-instability / instability
    ...     rng.normal(0.3, 0.05, 30),  # recovery
    ... ])
    >>> result = detect_regimes(np.abs(delta_phi))
    >>> set(result.labels).issubset({"stable", "pre-instability", "instability", "recovery"})
    True
    """
    delta_phi = np.asarray(delta_phi, dtype=float)
    if delta_phi.ndim != 1:
        raise ValueError(
            f"delta_phi must be a 1-D array; got shape {delta_phi.shape}."
        )
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(
            f"Quantile parameters must satisfy 0 ≤ q_low < q_high ≤ 1; "
            f"got q_low={q_low}, q_high={q_high}."
        )

    # ------------------------------------------------------------------
    # 1. Derive thresholds from the empirical quantile distribution of
    #    ΔΦ(t), consistent with the manuscript's quantile-based empirical
    #    characterisation (Section 5, Section 6).
    # ------------------------------------------------------------------
    theta_low = float(np.quantile(delta_phi, q_low))
    theta_high = float(np.quantile(delta_phi, q_high))

    # ------------------------------------------------------------------
    # 2. Assign initial three-regime labels element-wise.
    #
    #    Stable:          ΔΦ(t) < θ_low       (proximity to reference)
    #    Pre-instability: θ_low ≤ ΔΦ(t) < θ_high  (metastable, elevated)
    #    Instability:     ΔΦ(t) ≥ θ_high      (sustained deviation)
    # ------------------------------------------------------------------
    labels = np.empty(delta_phi.shape, dtype=object)
    labels[delta_phi < theta_low] = "stable"
    labels[(delta_phi >= theta_low) & (delta_phi < theta_high)] = "pre-instability"
    labels[delta_phi >= theta_high] = "instability"

    # ------------------------------------------------------------------
    # 3. Optional recovery detection (Section 6: "cellular recovery
    #    trajectories").
    #
    #    After the first instability time point, any subsequent point
    #    whose threshold-based label is "stable" (ΔΦ < θ_low) is
    #    relabelled "recovery": the system is returning toward the
    #    reference stability region following a departure.
    # ------------------------------------------------------------------
    if detect_recovery:
        instability_indices = np.where(labels == "instability")[0]
        if instability_indices.size > 0:
            first_instability = int(instability_indices[0])
            post_instability = np.arange(first_instability + 1, len(labels))
            recovery_mask = labels[post_instability] == "stable"
            labels[post_instability[recovery_mask]] = "recovery"

    return RegimeLabels(
        labels=labels,
        theta_low=theta_low,
        theta_high=theta_high,
    )
