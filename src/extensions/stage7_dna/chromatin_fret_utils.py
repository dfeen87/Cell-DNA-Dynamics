"""
chromatin_fret_utils – Low-level signal extraction for DNA-associated observables.

Provides three families of functions that convert raw instrument-level time
series into derived trajectories suitable for downstream ICE feature
extraction (see :mod:`dna_feature_mapping`):

1. **FRET intensity trajectories** – extract donor-channel, acceptor-channel,
   and FRET-efficiency traces from raw photon-count or fluorescence-intensity
   recordings.

2. **Chromatin mobility – MSD and coherence** – compute mean squared
   displacement (MSD) curves and rolling spatial-coherence estimates from
   locus-position time series.

3. **Repair-focus formation / dissolution dynamics** – derive rolling
   formation-rate and dissolution-rate traces from repair-focus fluorescence
   intensity recordings.

All functions operate exclusively on *time-resolved observables* (NumPy
arrays).  No biological interpretation is assigned beyond the raw-signal →
derived-trajectory transformation; the validity of any particular
interpretation must be evaluated empirically by the researcher.

Relation to the manuscript
--------------------------
Section 7 of the paper lists the following allowed input modalities:

* FRET time series             → :func:`extract_fret_trajectories`
* Chromatin mobility (loci)    → :func:`compute_msd`, :func:`compute_rolling_msd`,
                                  :func:`compute_msd_coherence`
* Repair-focus dynamics        → :func:`compute_focus_formation_rate`,
                                  :func:`compute_focus_dissolution_rate`,
                                  :func:`extract_repair_focus_dynamics`

The outputs of these functions are 1-D NumPy arrays of the same length *T*
as the input (with NaN for positions where a causal window is not yet full),
and can be passed directly to the feature-type functions in
:mod:`dna_feature_mapping`.

Usage example
-------------
::

    import numpy as np
    from src.extensions.stage7_dna.chromatin_fret_utils import (
        extract_fret_trajectories,
        compute_rolling_msd,
        extract_repair_focus_dynamics,
    )

    rng = np.random.default_rng(0)
    T = 500

    # --- FRET ---
    donor    = rng.uniform(0.5, 1.5, T)
    acceptor = rng.uniform(0.3, 1.2, T)
    fret = extract_fret_trajectories(donor, acceptor)
    # fret["donor"], fret["acceptor"], fret["efficiency"] are shape-(T,) arrays.

    # --- Chromatin mobility ---
    x_pos = np.cumsum(rng.normal(0, 0.05, T))
    msd_full    = compute_msd(x_pos, max_lag=50)                    # shape (50,)
    msd_rolling = compute_rolling_msd(x_pos, window_size=80, lag=5) # shape (T,)

    # --- Repair focus ---
    intensity = np.abs(rng.normal(0, 1, T))
    dynamics  = extract_repair_focus_dynamics(intensity, window_size=20)
    # dynamics["formation_rate"], dynamics["dissolution_rate"]  shape (T,)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_1d(arr: np.ndarray, name: str) -> np.ndarray:
    """Return *arr* as a float64 1-D array or raise ValueError."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"'{name}' must be a 1-D array; got shape {arr.shape}."
        )
    return arr


def _require_same_length(*named_arrays: tuple[np.ndarray, str]) -> int:
    """Verify all arrays share the same length and return it."""
    lengths = [len(arr) for arr, _ in named_arrays]
    if len(set(lengths)) != 1:
        detail = ", ".join(
            f"'{name}': {ln}" for (_, name), ln in zip(named_arrays, lengths)
        )
        raise ValueError(
            f"All input arrays must have the same length; got: {detail}."
        )
    return lengths[0]


# ---------------------------------------------------------------------------
# 1.  FRET intensity trajectories
# ---------------------------------------------------------------------------

def compute_fret_efficiency(
    donor: np.ndarray,
    acceptor: np.ndarray,
    *,
    gamma: float = 1.0,
    background_donor: float = 0.0,
    background_acceptor: float = 0.0,
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute the FRET efficiency time series from donor and acceptor channels.

    The apparent FRET efficiency is defined as:

        E(t) = I_A(t) / ( I_A(t) + gamma * I_D(t) )

    where ``I_D`` and ``I_A`` are the background-corrected donor and acceptor
    intensities, and ``gamma`` is a detector-correction factor (default 1.0
    for uncorrected instruments).  No molecular or structural interpretation
    is assigned to *E(t)*; it is treated as a dimensionless intensity-ratio
    trajectory.

    Parameters
    ----------
    donor:
        1-D array of donor-channel fluorescence intensities, shape ``(T,)``.
    acceptor:
        1-D array of acceptor-channel fluorescence intensities, shape ``(T,)``.
    gamma:
        Detector-correction factor.  Must be > 0.  Defaults to ``1.0``
        (no correction applied).
    background_donor:
        Scalar background offset subtracted from the donor channel before
        computing the ratio.  Defaults to ``0.0``.
    background_acceptor:
        Scalar background offset subtracted from the acceptor channel before
        computing the ratio.  Defaults to ``0.0``.
    eps:
        Small constant added to the denominator to prevent division by zero.
        Defaults to ``1e-10``.

    Returns
    -------
    np.ndarray of shape ``(T,)``
        FRET efficiency trajectory ``E(t)``.

    Raises
    ------
    ValueError
        If *donor* or *acceptor* are not 1-D arrays of the same length, or if
        *gamma* <= 0.
    """
    donor    = _require_1d(donor, "donor")
    acceptor = _require_1d(acceptor, "acceptor")
    _require_same_length((donor, "donor"), (acceptor, "acceptor"))

    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0; got {gamma}.")

    i_d = donor    - background_donor
    i_a = acceptor - background_acceptor
    denominator = i_a + gamma * i_d
    return i_a / (denominator + eps)


def extract_fret_trajectories(
    donor: np.ndarray,
    acceptor: np.ndarray,
    *,
    gamma: float = 1.0,
    background_donor: float = 0.0,
    background_acceptor: float = 0.0,
    eps: float = 1e-10,
) -> dict[str, np.ndarray]:
    """Extract FRET intensity trajectories from raw donor and acceptor channels.

    Returns three derived trajectories:

    * ``"donor"``      – background-corrected donor intensity ``I_D(t)``.
    * ``"acceptor"``   – background-corrected acceptor intensity ``I_A(t)``.
    * ``"efficiency"`` – FRET efficiency ``E(t) = I_A / (I_A + gamma * I_D)``.

    All three arrays have shape ``(T,)`` and dtype float64.  No biological
    interpretation is assigned beyond the observable -> derived-trajectory
    mapping.

    Parameters
    ----------
    donor:
        1-D donor-channel intensity time series, shape ``(T,)``.
    acceptor:
        1-D acceptor-channel intensity time series, shape ``(T,)``.
    gamma:
        Detector-correction factor (see :func:`compute_fret_efficiency`).
        Must be > 0.
    background_donor:
        Scalar background offset for the donor channel.
    background_acceptor:
        Scalar background offset for the acceptor channel.
    eps:
        Small constant guarding against division by zero.

    Returns
    -------
    dict with keys ``"donor"``, ``"acceptor"``, ``"efficiency"``
        Each value is a 1-D NumPy array of shape ``(T,)``.
    """
    donor    = _require_1d(donor, "donor")
    acceptor = _require_1d(acceptor, "acceptor")
    _require_same_length((donor, "donor"), (acceptor, "acceptor"))

    i_d = donor    - background_donor
    i_a = acceptor - background_acceptor
    efficiency = compute_fret_efficiency(
        donor, acceptor,
        gamma=gamma,
        background_donor=background_donor,
        background_acceptor=background_acceptor,
        eps=eps,
    )

    return {
        "donor":      i_d,
        "acceptor":   i_a,
        "efficiency": efficiency,
    }


# ---------------------------------------------------------------------------
# 2.  Chromatin mobility – MSD and coherence
# ---------------------------------------------------------------------------

def compute_msd(
    positions: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    """Compute the ensemble-averaged mean squared displacement (MSD) curve.

    For a 1-D position time series ``x(t)`` the MSD at lag ``tau`` is:

        MSD(tau) = (1 / (T - tau)) * sum_{t=0}^{T-tau-1} [ x(t + tau) - x(t) ]^2

    No physical model (Brownian, anomalous diffusion, etc.) is fitted; the
    function returns the raw MSD values for lags ``tau = 1, 2, ..., max_lag``.

    Parameters
    ----------
    positions:
        1-D locus-position time series, shape ``(T,)``.
    max_lag:
        Maximum lag (in samples) to compute.  Must satisfy
        ``1 <= max_lag < T``.

    Returns
    -------
    np.ndarray of shape ``(max_lag,)``
        MSD values for lags ``1`` through ``max_lag``.  Index ``k``
        corresponds to lag ``tau = k + 1``.

    Raises
    ------
    ValueError
        If *positions* is not 1-D or *max_lag* is out of range.
    """
    positions = _require_1d(positions, "positions")
    T = len(positions)
    if not (1 <= max_lag < T):
        raise ValueError(
            f"max_lag must satisfy 1 <= max_lag < T={T}; got max_lag={max_lag}."
        )

    msd = np.empty(max_lag, dtype=float)
    for lag in range(1, max_lag + 1):
        displacements = positions[lag:] - positions[:-lag]
        msd[lag - 1] = float(np.mean(displacements ** 2))
    return msd


def compute_rolling_msd(
    positions: np.ndarray,
    window_size: int,
    lag: int,
) -> np.ndarray:
    """Compute a rolling MSD at a fixed lag within a causal sliding window.

    For each time point *t* (once the window is full), a local MSD at the
    specified *lag* is computed over the window ``[t - window_size + 1, t]``:

        MSD_local(t, tau) = (1 / (W - tau)) * sum_{k} [ x(k + tau) - x(k) ]^2

    where the sum runs over the ``W - tau`` available displacement pairs within
    the window.  The first ``window_size - 1`` output samples are NaN (causal
    window).

    Parameters
    ----------
    positions:
        1-D locus-position time series, shape ``(T,)``.
    window_size:
        Rolling window length in samples.  Must satisfy
        ``window_size > lag + 1``.
    lag:
        Fixed lag ``tau`` (in samples) at which the MSD is evaluated within
        each window.  Must satisfy ``1 <= lag < window_size``.

    Returns
    -------
    np.ndarray of shape ``(T,)``
        Rolling MSD time series with NaN for the first ``window_size - 1``
        samples.

    Raises
    ------
    ValueError
        If inputs are inconsistent.
    """
    positions = _require_1d(positions, "positions")
    T = len(positions)

    if not (1 <= lag < window_size):
        raise ValueError(
            f"lag must satisfy 1 <= lag < window_size={window_size}; "
            f"got lag={lag}."
        )
    if window_size > T:
        raise ValueError(
            f"window_size={window_size} exceeds series length T={T}."
        )

    out = np.full(T, np.nan)
    for t in range(window_size - 1, T):
        window = positions[t - window_size + 1 : t + 1]
        displacements = window[lag:] - window[:-lag]
        out[t] = float(np.mean(displacements ** 2))
    return out


def compute_msd_coherence(
    positions_2d: np.ndarray,
    window_size: int,
    lag: int,
) -> np.ndarray:
    """Compute rolling spatial coherence from multi-locus MSD trajectories.

    The coherence is defined as the mean pairwise Pearson correlation of the
    per-locus rolling-MSD time series computed at the specified *lag*.  A high
    value indicates that multiple loci share similar displacement dynamics
    within the observation window (coordinated chromatin motion).

    Parameters
    ----------
    positions_2d:
        2-D array of shape ``(n_loci, T)`` with ``n_loci >= 2``, where each
        row is a locus-position time series.
    window_size:
        Rolling window length passed to :func:`compute_rolling_msd`.
    lag:
        MSD lag passed to :func:`compute_rolling_msd`.

    Returns
    -------
    np.ndarray of shape ``(T,)``
        Rolling MSD-coherence time series with NaN for the first
        ``window_size - 1`` samples.  Values are in ``[-1, 1]``.

    Raises
    ------
    ValueError
        If *positions_2d* is not a 2-D array with at least 2 rows, or if
        *window_size* / *lag* constraints are violated.
    """
    positions_2d = np.asarray(positions_2d, dtype=float)
    if positions_2d.ndim != 2 or positions_2d.shape[0] < 2:
        raise ValueError(
            "positions_2d must be a 2-D array of shape (n_loci, T) "
            f"with n_loci >= 2; got shape {positions_2d.shape}."
        )

    n_loci, T = positions_2d.shape

    # Compute per-locus rolling MSD series
    msd_series = np.stack(
        [compute_rolling_msd(positions_2d[i], window_size, lag) for i in range(n_loci)],
        axis=0,
    )  # shape (n_loci, T)

    # Mean pairwise Pearson r across all locus pairs (using all valid samples)
    out = np.full(T, np.nan)
    for t in range(window_size - 1, T):
        # Use accumulated MSD values up to t (only valid ones)
        r_values: list[float] = []
        for i in range(n_loci):
            for j in range(i + 1, n_loci):
                a = msd_series[i, :t + 1]
                b = msd_series[j, :t + 1]
                mask = np.isfinite(a) & np.isfinite(b)
                if mask.sum() < 2:
                    continue
                std_a = float(np.std(a[mask]))
                std_b = float(np.std(b[mask]))
                if std_a < 1e-12 or std_b < 1e-12:
                    r_values.append(0.0)
                else:
                    r_values.append(float(np.corrcoef(a[mask], b[mask])[0, 1]))
        if r_values:
            out[t] = float(np.mean(r_values))
    return out


# ---------------------------------------------------------------------------
# 3.  Repair-focus formation / dissolution dynamics
# ---------------------------------------------------------------------------

def compute_focus_formation_rate(
    intensity: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Compute a rolling repair-focus formation rate from intensity dynamics.

    The formation rate is the mean of the *positive* first-differences within
    each causal window:

        formation_rate(t) = mean( max(delta_x_k, 0) )   for k in window

    where ``delta_x_k = x(k) - x(k-1)`` are the intensity increments.  This
    quantity captures the average rate of intensity increase within the window,
    corresponding to focus assembly events.  No biological interpretation is
    assigned beyond this observable definition.

    Parameters
    ----------
    intensity:
        1-D repair-focus fluorescence intensity time series, shape ``(T,)``.
    window_size:
        Rolling window length in samples (>= 2 required for differences).

    Returns
    -------
    np.ndarray of shape ``(T,)``
        Rolling formation-rate time series with NaN for the first
        ``window_size - 1`` samples.

    Raises
    ------
    ValueError
        If *intensity* is not 1-D or *window_size* < 2.
    """
    intensity = _require_1d(intensity, "intensity")
    if window_size < 2:
        raise ValueError(
            f"window_size must be >= 2; got {window_size}."
        )

    T = len(intensity)
    out = np.full(T, np.nan)
    for t in range(window_size - 1, T):
        window = intensity[t - window_size + 1 : t + 1]
        diffs = np.diff(window)
        out[t] = float(np.mean(np.maximum(diffs, 0.0)))
    return out


def compute_focus_dissolution_rate(
    intensity: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Compute a rolling repair-focus dissolution rate from intensity dynamics.

    The dissolution rate is the mean of the magnitudes of *negative*
    first-differences within each causal window:

        dissolution_rate(t) = mean( max(-delta_x_k, 0) )   for k in window

    where ``delta_x_k = x(k) - x(k-1)`` are the intensity increments.  This
    quantity captures the average rate of intensity decrease, corresponding to
    focus disassembly events.  No biological interpretation is assigned beyond
    this observable definition.

    Parameters
    ----------
    intensity:
        1-D repair-focus fluorescence intensity time series, shape ``(T,)``.
    window_size:
        Rolling window length in samples (>= 2 required for differences).

    Returns
    -------
    np.ndarray of shape ``(T,)``
        Rolling dissolution-rate time series with NaN for the first
        ``window_size - 1`` samples.

    Raises
    ------
    ValueError
        If *intensity* is not 1-D or *window_size* < 2.
    """
    intensity = _require_1d(intensity, "intensity")
    if window_size < 2:
        raise ValueError(
            f"window_size must be >= 2; got {window_size}."
        )

    T = len(intensity)
    out = np.full(T, np.nan)
    for t in range(window_size - 1, T):
        window = intensity[t - window_size + 1 : t + 1]
        diffs = np.diff(window)
        out[t] = float(np.mean(np.maximum(-diffs, 0.0)))
    return out


def extract_repair_focus_dynamics(
    intensity: np.ndarray,
    window_size: int,
) -> dict[str, np.ndarray]:
    """Extract formation and dissolution rate trajectories from focus intensity.

    Combines :func:`compute_focus_formation_rate` and
    :func:`compute_focus_dissolution_rate` into a single convenience function.

    Parameters
    ----------
    intensity:
        1-D repair-focus fluorescence intensity time series, shape ``(T,)``.
    window_size:
        Rolling window length in samples (>= 2).

    Returns
    -------
    dict with keys ``"formation_rate"`` and ``"dissolution_rate"``
        Each value is a 1-D NumPy array of shape ``(T,)`` with NaN for the
        first ``window_size - 1`` samples.
    """
    return {
        "formation_rate":   compute_focus_formation_rate(intensity, window_size),
        "dissolution_rate": compute_focus_dissolution_rate(intensity, window_size),
    }
