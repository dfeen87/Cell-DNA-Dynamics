"""
Segmentation utilities – Stage 5 Regime-Transition Detection
=============================================================

Provides three helper utilities that support the manuscript's validation
pipeline:

1. :func:`compute_threshold` – empirically derive the critical threshold Φ_c
   from an instability-magnitude series ΔΦ(t) (manuscript Section 5).

2. :func:`regime_shading_intervals` – convert per-time-point regime labels into
   contiguous (start, end, regime) intervals suitable for axis shading
   (e.g. ``matplotlib.axes.Axes.axvspan``).

3. :func:`extract_transition_events` – extract the ordered sequence of regime
   transitions from the label array, annotated with the time index at which
   each transition occurs.

None of the functions assign biological meaning to the detected segments; they
operate purely on the numerical ΔΦ(t) series and the discrete label array
produced by :func:`~src.stage5_regime_transitions.detect_regimes.detect_regimes`.

References
----------
Manuscript Section 5 (Regime Transition Structure):

    "The threshold Φ_c is not assumed to be universal and must be determined
     empirically for each system under study, for example through baseline
     characterisation, surrogate analysis, or controlled perturbation
     experiments."

Manuscript Section 8.3 (Synthetic Validation Protocol):

    "Baseline statistics (μ_X, σ_X) are computed over the stable regime
     (t < 60 s)."
"""

from __future__ import annotations

from typing import List, NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Threshold computation
# ---------------------------------------------------------------------------

def compute_threshold(
    delta_phi: np.ndarray,
    method: str = "quantile",
    *,
    q: float = 0.90,
    baseline_mask: np.ndarray | None = None,
    n_sigma: float = 2.0,
) -> float:
    """Compute an empirical threshold Φ_c from the instability magnitude ΔΦ(t).

    The threshold is *not* assumed to be universal (manuscript Section 5).
    Two estimation strategies are supported:

    ``"quantile"``
        Φ_c = quantile(ΔΦ, q).  The quantile is computed over the full
        supplied series unless a *baseline_mask* is provided, in which case
        only the masked samples are used.  This matches the quantile-based
        empirical characterisation described in Sections 5 and 6 of the
        manuscript.

    ``"baseline"``
        Φ_c = μ_baseline + n_sigma · σ_baseline, where μ and σ are the mean
        and standard deviation of ΔΦ(t) restricted to the stable baseline
        window identified by *baseline_mask*.  This mirrors the baseline
        normalisation defined in manuscript Eq. (27):

            X_n(t) = ( X(t) − μ_X ) / σ_X,   X ∈ {E, I, C}

        and the protocol in Section 8.3 which computes baseline statistics
        "over the stable regime (t < 60 s)".

    Parameters
    ----------
    delta_phi : array_like, shape (T,)
        Instability magnitude ΔΦ(t) ≥ 0 as produced by
        :func:`~src.stage5_regime_transitions.compute_delta_phi.compute_delta_phi`.
    method : {"quantile", "baseline"}, default "quantile"
        Threshold estimation strategy (see above).
    q : float, default 0.90
        Quantile level for the ``"quantile"`` method.  Must be in (0, 1].
    baseline_mask : array_like of bool, shape (T,), optional
        Boolean mask identifying the stable baseline samples.
        - For ``"quantile"``: when provided, the quantile is computed only
          over the masked (``True``) samples.
        - For ``"baseline"``: required; raises ``ValueError`` if omitted.
    n_sigma : float, default 2.0
        Number of standard deviations above the baseline mean for the
        ``"baseline"`` method.  Must be non-negative.

    Returns
    -------
    threshold : float
        Empirical threshold Φ_c ≥ 0.

    Raises
    ------
    ValueError
        If *delta_phi* is not 1-D, *method* is unrecognised, quantile
        parameters are out of range, or the ``"baseline"`` method is called
        without a valid *baseline_mask*.

    Examples
    --------
    >>> import numpy as np
    >>> from src.stage5_regime_transitions.segmentation_utils import compute_threshold
    >>> rng = np.random.default_rng(0)
    >>> delta_phi = np.abs(rng.normal(1.0, 0.5, 300))
    >>> thr = compute_threshold(delta_phi, method="quantile", q=0.90)
    >>> float(thr) >= 0
    True
    >>> mask = np.zeros(300, dtype=bool)
    >>> mask[:60] = True  # first 60 samples are the stable baseline
    >>> thr_b = compute_threshold(delta_phi, method="baseline", baseline_mask=mask, n_sigma=2.0)
    >>> float(thr_b) >= 0
    True
    """
    delta_phi = np.asarray(delta_phi, dtype=float)
    if delta_phi.ndim != 1:
        raise ValueError(
            f"delta_phi must be a 1-D array; got shape {delta_phi.shape}."
        )

    if method == "quantile":
        if not (0.0 < q <= 1.0):
            raise ValueError(
                f"q must be in (0, 1]; got q={q}."
            )
        samples = delta_phi
        if baseline_mask is not None:
            mask = np.asarray(baseline_mask, dtype=bool)
            if mask.shape != delta_phi.shape:
                raise ValueError(
                    "baseline_mask must have the same shape as delta_phi; "
                    f"got {mask.shape} vs {delta_phi.shape}."
                )
            samples = delta_phi[mask]
            if samples.size == 0:
                raise ValueError(
                    "baseline_mask selects no samples; cannot compute quantile."
                )
        return float(np.quantile(samples, q))

    elif method == "baseline":
        if baseline_mask is None:
            raise ValueError(
                "baseline_mask is required for method='baseline'."
            )
        if n_sigma < 0.0:
            raise ValueError(
                f"n_sigma must be non-negative; got n_sigma={n_sigma}."
            )
        mask = np.asarray(baseline_mask, dtype=bool)
        if mask.shape != delta_phi.shape:
            raise ValueError(
                "baseline_mask must have the same shape as delta_phi; "
                f"got {mask.shape} vs {delta_phi.shape}."
            )
        baseline_samples = delta_phi[mask]
        if baseline_samples.size == 0:
            raise ValueError(
                "baseline_mask selects no samples; cannot compute baseline statistics."
            )
        mu = float(np.mean(baseline_samples))
        sigma = float(np.std(baseline_samples, ddof=0))
        return mu + n_sigma * sigma

    else:
        raise ValueError(
            f"Unknown method '{method}'. Supported methods: 'quantile', 'baseline'."
        )


# ---------------------------------------------------------------------------
# 2. Regime shading intervals
# ---------------------------------------------------------------------------

class ShadingInterval(NamedTuple):
    """A contiguous block of time steps assigned to the same regime.

    Attributes
    ----------
    regime : str
        Regime label (one of ``"stable"``, ``"pre-instability"``,
        ``"instability"``, ``"recovery"``).
    start : int
        Index of the first time step in the interval (inclusive).
    stop : int
        Index one past the last time step in the interval (exclusive),
        matching Python slice semantics.  The interval covers indices
        ``[start, stop)``.
    """

    regime: str
    start: int
    stop: int


def regime_shading_intervals(labels: np.ndarray) -> List[ShadingInterval]:
    """Convert a per-time-point regime label array into contiguous intervals.

    Groups consecutive identical labels into :class:`ShadingInterval` records
    that can be passed directly to ``matplotlib.axes.Axes.axvspan`` (or any
    interval-based shading routine) for visualization of regime boundaries in
    the validation pipeline.

    Parameters
    ----------
    labels : array_like, shape (T,), dtype str or object
        Per-time-point regime strings as produced by
        :func:`~src.stage5_regime_transitions.detect_regimes.detect_regimes`.
        Each element must be one of ``"stable"``, ``"pre-instability"``,
        ``"instability"``, or ``"recovery"``.

    Returns
    -------
    intervals : list of :class:`ShadingInterval`
        Ordered list of non-overlapping contiguous regime intervals that
        together cover the full index range ``[0, T)``.  The list is ordered
        by increasing *start* index.

    Raises
    ------
    ValueError
        If *labels* is not a 1-D array or is empty.

    Examples
    --------
    >>> import numpy as np
    >>> from src.stage5_regime_transitions.segmentation_utils import regime_shading_intervals
    >>> labels = np.array(
    ...     ["stable"] * 4 + ["pre-instability"] * 3 + ["instability"] * 2
    ... )
    >>> intervals = regime_shading_intervals(labels)
    >>> [(iv.regime, iv.start, iv.stop) for iv in intervals]
    [('stable', 0, 4), ('pre-instability', 4, 7), ('instability', 7, 9)]
    """
    labels = np.asarray(labels, dtype=object)
    if labels.ndim != 1:
        raise ValueError(
            f"labels must be a 1-D array; got shape {labels.shape}."
        )
    if labels.size == 0:
        raise ValueError("labels array must not be empty.")

    intervals: List[ShadingInterval] = []
    current_regime = labels[0]
    block_start = 0

    for i in range(1, len(labels)):
        if labels[i] != current_regime:
            intervals.append(ShadingInterval(str(current_regime), block_start, i))
            current_regime = labels[i]
            block_start = i

    # Close the final block.
    intervals.append(ShadingInterval(str(current_regime), block_start, len(labels)))

    return intervals


# ---------------------------------------------------------------------------
# 3. Transition event extraction
# ---------------------------------------------------------------------------

class TransitionEvent(NamedTuple):
    """A single regime-transition event.

    Attributes
    ----------
    from_regime : str
        Regime label at time step ``index - 1``.
    to_regime : str
        Regime label at time step ``index`` (the first step of the new regime).
    index : int
        Time-step index at which the transition occurs, i.e. the first index
        that belongs to the new regime.
    """

    from_regime: str
    to_regime: str
    index: int


def extract_transition_events(labels: np.ndarray) -> List[TransitionEvent]:
    """Extract regime-transition events from a per-time-point label array.

    A transition event is recorded whenever adjacent time steps carry
    different regime labels.  The returned list contains one
    :class:`TransitionEvent` per detected boundary, ordered by increasing
    *index*.

    This function supports the manuscript's validation pipeline (Section 5
    and 8.3) by providing a discrete, ordered representation of when the
    instability magnitude ΔΦ(t) crosses regime boundaries—enabling
    quantitative evaluation of detection lead times, false-positive rates,
    and regime-ordering statistics (manuscript Table 3).

    Parameters
    ----------
    labels : array_like, shape (T,), dtype str or object
        Per-time-point regime strings as produced by
        :func:`~src.stage5_regime_transitions.detect_regimes.detect_regimes`.

    Returns
    -------
    events : list of :class:`TransitionEvent`
        Ordered list of transition events.  Empty if *labels* contains only
        a single regime throughout.

    Raises
    ------
    ValueError
        If *labels* is not a 1-D array or is empty.

    Examples
    --------
    >>> import numpy as np
    >>> from src.stage5_regime_transitions.segmentation_utils import extract_transition_events
    >>> labels = np.array(
    ...     ["stable"] * 4 + ["pre-instability"] * 3 + ["instability"] * 2
    ... )
    >>> events = extract_transition_events(labels)
    >>> [(e.from_regime, e.to_regime, e.index) for e in events]
    [('stable', 'pre-instability', 4), ('pre-instability', 'instability', 7)]
    """
    labels = np.asarray(labels, dtype=object)
    if labels.ndim != 1:
        raise ValueError(
            f"labels must be a 1-D array; got shape {labels.shape}."
        )
    if labels.size == 0:
        raise ValueError("labels array must not be empty.")

    events: List[TransitionEvent] = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            events.append(
                TransitionEvent(
                    from_regime=str(labels[i - 1]),
                    to_regime=str(labels[i]),
                    index=i,
                )
            )
    return events
