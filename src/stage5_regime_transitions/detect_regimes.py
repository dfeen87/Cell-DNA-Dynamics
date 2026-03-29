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


def _enforce_min_dwell(labels: np.ndarray, min_dwell: int) -> np.ndarray:
    """Merge regime intervals shorter than *min_dwell* samples into neighbours.

    Iteratively finds the first contiguous run whose length is less than
    *min_dwell* and reassigns it to match the longer of its two adjacent runs
    (left neighbour wins a tie).  The loop terminates once all runs meet the
    minimum length requirement.

    Parameters
    ----------
    labels : np.ndarray, shape (T,), dtype object
        Per-time-point regime label array to stabilise in-place copy.
    min_dwell : int
        Minimum number of consecutive samples required to keep a regime run.
        Values ≤ 1 disable the filter.

    Returns
    -------
    np.ndarray
        A new label array with all runs at least *min_dwell* samples long.
    """
    if min_dwell <= 1:
        return labels.copy()

    result = labels.copy()

    while True:
        # Build run-length encoding: list of [regime, start, stop].
        runs: list[list] = []
        cur = result[0]
        start = 0
        for i in range(1, len(result)):
            if result[i] != cur:
                runs.append([cur, start, i])
                cur = result[i]
                start = i
        runs.append([cur, start, len(result)])

        # Find the first run that is too short.
        short_idx: int | None = None
        for k, (_, s, e) in enumerate(runs):
            if (e - s) < min_dwell:
                short_idx = k
                break

        if short_idx is None:
            break  # All runs satisfy the minimum dwell time.

        k = short_idx
        left_len = (runs[k - 1][2] - runs[k - 1][1]) if k > 0 else -1
        right_len = (runs[k + 1][2] - runs[k + 1][1]) if k < len(runs) - 1 else -1

        if left_len >= right_len and k > 0:
            # Absorb into the left neighbour.
            result[runs[k][1] : runs[k][2]] = runs[k - 1][0]
        elif k < len(runs) - 1:
            # Absorb into the right neighbour.
            result[runs[k][1] : runs[k][2]] = runs[k + 1][0]
        else:
            # Single-run array; nothing to merge.
            break

    return result


def detect_regimes(
    delta_phi: np.ndarray,
    q_low: float = 0.75,
    q_high: float = 0.90,
    detect_recovery: bool = True,
    min_dwell: int = 30,
    hysteresis: float = 0.05,
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

    Hysteresis (*hysteresis* > 0):
    To suppress rapid oscillations when ΔΦ(t) hovers near a boundary, the
    classifier uses offset exit thresholds.  The *enter* threshold for each
    regime is the unmodified quantile value; the *exit* threshold is lowered
    by a fraction *hysteresis* of the inter-threshold range
    (θ_high − θ_low):

    - The classifier leaves ``"pre-instability"`` back to ``"stable"`` only
      when ΔΦ(t) < θ_low − δ  (δ = (θ_high − θ_low) × hysteresis).
    - The classifier leaves ``"instability"`` back to ``"pre-instability"``
      only when ΔΦ(t) < θ_high − δ.

    This prevents boundary-hovering signals from generating spurious
    short-lived regime flips (label jitter).

    Optional recovery detection (when *detect_recovery* is ``True``):
    after the first time point at which ``"instability"`` is assigned,
    any subsequent time point whose raw label would be ``"stable"``
    (i.e. ΔΦ(t) < θ_low) is instead labelled ``"recovery"``, reflecting
    the system's return toward the reference stability region (Section 6:
    "cellular recovery trajectories").

    Minimum dwell-time stabilisation (*min_dwell* > 1):
    After hysteresis labelling and optional recovery detection, any
    contiguous run of identical labels shorter than *min_dwell* samples is
    merged into its longer adjacent run.  This ensures the segmentation
    output contains broad, stable regime bands rather than rapid
    oscillations around threshold boundaries.

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
    min_dwell : int, default 30
        Minimum number of consecutive samples required to retain a regime
        run.  Any run shorter than this value is absorbed into its longer
        neighbour.  Set to 1 to disable minimum-dwell-time stabilisation.
    hysteresis : float, default 0.05
        Hysteresis fraction applied to exit thresholds to suppress label
        jitter near regime boundaries.  The exit threshold for each
        boundary is offset downward by ``hysteresis × (θ_high − θ_low)``.
        Set to 0.0 to disable hysteresis.

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
    # 2. Assign regime labels via a stateful hysteresis scan.
    #
    #    Enter thresholds match the unmodified quantile values.
    #    Exit thresholds are offset downward by a fraction of the
    #    inter-threshold range to prevent boundary hovering from generating
    #    spurious short-lived regime flips (label jitter).
    #
    #    Stable:          ΔΦ(t) < θ_low       (enter)
    #                     ΔΦ(t) < θ_low_exit  (exit pre-instability)
    #    Pre-instability: θ_low ≤ ΔΦ(t)       (enter from stable)
    #                     θ_high_exit ≤ ΔΦ(t) (remain after leaving instability)
    #    Instability:     ΔΦ(t) ≥ θ_high      (enter)
    # ------------------------------------------------------------------
    delta_band = (theta_high - theta_low) * float(hysteresis)
    theta_low_exit = theta_low - delta_band    # exit pre-instability → stable
    theta_high_exit = theta_high - delta_band  # exit instability → pre-instability

    labels = np.empty(delta_phi.shape, dtype=object)

    # Initialise state from the first sample (no hysteresis on entry).
    if delta_phi[0] >= theta_high:
        state: str = "instability"
    elif delta_phi[0] >= theta_low:
        state = "pre-instability"
    else:
        state = "stable"

    for i, dp in enumerate(delta_phi):
        if state == "stable":
            if dp >= theta_high:
                state = "instability"
            elif dp >= theta_low:
                state = "pre-instability"
        elif state == "pre-instability":
            if dp >= theta_high:
                state = "instability"
            elif dp < theta_low_exit:
                state = "stable"
        else:  # state == "instability"
            if dp < theta_high_exit:
                state = "pre-instability"
        labels[i] = state

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

    # ------------------------------------------------------------------
    # 4. Minimum dwell-time stabilisation.
    #
    #    Merge short regime runs to prevent the classifier from toggling
    #    on single-sample threshold crossings due to measurement noise.
    # ------------------------------------------------------------------
    if min_dwell > 1:
        labels = _enforce_min_dwell(labels, min_dwell)

    return RegimeLabels(
        labels=labels,
        theta_low=theta_low,
        theta_high=theta_high,
    )
