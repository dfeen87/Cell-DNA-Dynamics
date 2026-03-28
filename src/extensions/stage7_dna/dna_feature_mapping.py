"""
dna_feature_mapping – Observable-to-ICE component mapping for DNA-associated
dynamics.

Implements the explicit mapping from DNA-associated time-resolved observables
to the three triadic ICE components (Energy, Information, Coherence) as
described in Section 7 and Table 2 of the manuscript.

Mapping rules (manuscript Section 7, Table 2):

  E – Energy-linked DNA observables
      Representative observables: ATP dependence of repair dynamics,
      FRET efficiency as a conformational-energy proxy, replication-stress
      marker intensity.
      Supported feature_type values: 'fret_energy_proxy',
      'repair_intensity_rms', 'replication_stress_rms'.

  I – Information-linked DNA observables
      Representative observables: transcription burst variability,
      chromatin accessibility heterogeneity.
      Supported feature_type values: 'burst_variability',
      'chromatin_entropy', 'burst_permutation_entropy'.

  C – Coherence-linked DNA observables
      Representative observables: repair focus synchronization, locus
      co-motion.
      Supported feature_type values: 'locus_co_motion', 'focus_synchrony',
      'mobility_phase_coherence'.

Key constraint (manuscript Section 7): static DNA sequences are not modeled
directly.  All five allowed input modalities are *time-resolved* signals:

  * chromatin mobility trajectories  (locus-position time series)
  * FRET time series                 (efficiency or donor/acceptor traces)
  * transcription burst traces       (single-cell RNA / reporter intensity)
  * repair-focus dynamics            (fluorescence intensity per focus)
  * replication-stress trajectories  (marker intensity vs. time)

No biological interpretation is assigned beyond the caller-specified
observable → ICE component mapping.  The validity of any particular
mapping must be evaluated empirically by the researcher.

Usage example
-------------
::

    import numpy as np
    from src.extensions.stage7_dna.dna_feature_mapping import (
        DNAFeatureSpec,
        dna_feature_mapping,
    )

    rng = np.random.default_rng(42)
    T = 500

    raw_data = {
        "fret_eff":    rng.uniform(0.2, 0.8, T),   # FRET efficiency trace
        "burst_trace": rng.exponential(1.0, T),     # transcription burst
        "locus_x1":   np.cumsum(rng.normal(0, 0.1, T)),
        "locus_x2":   np.cumsum(rng.normal(0, 0.1, T)),
    }

    specs = [
        DNAFeatureSpec("fret_rms",  "E", "fret_energy_proxy",  ["fret_eff"],              window_size=50),
        DNAFeatureSpec("burst_cv",  "I", "burst_variability",  ["burst_trace"],           window_size=50),
        DNAFeatureSpec("co_motion", "C", "locus_co_motion",    ["locus_x1", "locus_x2"], window_size=50),
    ]

    ice = dna_feature_mapping(raw_data, specs, baseline_window=(0, 150))
    # ice.delta_E, ice.delta_I, ice.delta_C are shape-(T,) normalized arrays.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from itertools import combinations, permutations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.signal import hilbert

# ---------------------------------------------------------------------------
# Ensure repository root is importable regardless of the working directory.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.models import ICEStateSeries  # noqa: E402


# ---------------------------------------------------------------------------
# Observable catalogue
# ---------------------------------------------------------------------------

#: Maps ICE component → {feature_type: description}.
#: Derived directly from Table 2 and Section 7 of the manuscript.
DNA_OBSERVABLE_CATALOGUE: dict[str, dict[str, str]] = {
    "E": {
        "fret_energy_proxy": (
            "Rolling RMS of a FRET efficiency (or donor/acceptor intensity) "
            "time series.  FRET efficiency reflects inter-molecular distance "
            "and is used here as a conformational-energy proxy for DNA "
            "dynamics (Table 2, Energy-linked).  Implements Eq. (24)."
        ),
        "repair_intensity_rms": (
            "Rolling RMS of a repair-focus fluorescence intensity trace.  "
            "Repair-focus amplitude is coupled to ATP-dependent repair "
            "machinery, making it an energy proxy for DNA-damage response "
            "dynamics (Table 2, Energy-linked).  Implements Eq. (24)."
        ),
        "replication_stress_rms": (
            "Rolling RMS of a replication-stress marker intensity trajectory.  "
            "Stress-marker amplitude reflects energy demands at stalled or "
            "active replication forks (Table 2, Stress-linked, mapped to E "
            "via fork-activity proxy).  Implements Eq. (24)."
        ),
    },
    "I": {
        "burst_variability": (
            "Rolling coefficient of variation (σ / |μ|) of a transcription "
            "burst trace.  High CV corresponds to high burst heterogeneity "
            "and gene-regulatory diversity (Table 2, Information-linked: "
            "'transcription burst variability')."
        ),
        "chromatin_entropy": (
            "Rolling spectral Shannon entropy (Eq. 25) of a chromatin "
            "mobility or accessibility signal.  Entropy quantifies chromatin "
            "conformational heterogeneity (Table 2, Information-linked: "
            "'chromatin accessibility heterogeneity')."
        ),
        "burst_permutation_entropy": (
            "Rolling normalised permutation entropy of a transcription burst "
            "trace.  Permutation entropy captures ordinal-pattern complexity "
            "and complements CV as a burst-variability measure (Table 2, "
            "Information-linked)."
        ),
    },
    "C": {
        "locus_co_motion": (
            "Rolling mean pairwise Pearson correlation across ≥ 2 locus "
            "mobility (position) time series.  Co-motion of genomic loci "
            "reflects coherent chromatin reorganisation (Table 2, "
            "Coherence-linked: 'locus co-motion')."
        ),
        "focus_synchrony": (
            "Rolling mean pairwise Pearson correlation across ≥ 2 repair-focus "
            "intensity channels.  Synchronous recruitment or dissolution of "
            "repair foci is a coherence signal (Table 2, Coherence-linked: "
            "'repair focus synchronization')."
        ),
        "mobility_phase_coherence": (
            "Rolling phase-coherence proxy (Eq. 26) of a single chromatin "
            "locus mobility trace: C = 1 / (Var(Δφ) + ε).  Stable phase "
            "increments indicate coherent locus motion within the nucleus."
        ),
    },
}

# Flat set of all valid (feature_type, ice_component) pairs.
_VALID_ASSIGNMENTS: frozenset[tuple[str, str]] = frozenset(
    (ft, comp)
    for comp, ftypes in DNA_OBSERVABLE_CATALOGUE.items()
    for ft in ftypes
)

# Feature types that accept ≥ 2 channels (all others accept exactly 1).
_MULTI_CHANNEL_TYPES: frozenset[str] = frozenset({"locus_co_motion", "focus_synchrony"})


# ---------------------------------------------------------------------------
# DNAFeatureSpec
# ---------------------------------------------------------------------------

@dataclass
class DNAFeatureSpec:
    """Declare one DNA-associated feature to extract and its ICE assignment.

    Parameters
    ----------
    name:
        Unique output name for the computed feature time series.  This name
        will appear as a key in the *observables* dict returned by
        :func:`map_dna_to_ice`.

    ice_component:
        ICE component this feature is assigned to.  Must be one of
        ``"E"``, ``"I"``, or ``"C"``.

    feature_type:
        Algorithm used to extract the feature.  Must be one of the keys
        listed in :data:`DNA_OBSERVABLE_CATALOGUE` for the given
        *ice_component*.  Allowed values:

        * ``"E"``: ``'fret_energy_proxy'``, ``'repair_intensity_rms'``,
          ``'replication_stress_rms'``.
        * ``"I"``: ``'burst_variability'``, ``'chromatin_entropy'``,
          ``'burst_permutation_entropy'``.
        * ``"C"``: ``'locus_co_motion'``, ``'focus_synchrony'``,
          ``'mobility_phase_coherence'``.

    channels:
        Names of the raw-data channels (keys in the *data* dict passed to
        :func:`map_dna_to_ice`) required by this feature.

        * Single-channel types (exactly one channel required):
          ``'fret_energy_proxy'``, ``'repair_intensity_rms'``,
          ``'replication_stress_rms'``, ``'burst_variability'``,
          ``'chromatin_entropy'``, ``'burst_permutation_entropy'``,
          ``'mobility_phase_coherence'``.
        * Multi-channel types (two or more channels required):
          ``'locus_co_motion'``, ``'focus_synchrony'``.

    window_size:
        Rolling window length in samples.  Must be ≥ 2.

    kwargs:
        Additional keyword arguments forwarded to the underlying feature
        function.  Supported overrides:

        * ``'burst_variability'``: ``eps`` (float, default ``1e-10``) –
          added to ``|μ|`` before division.
        * ``'chromatin_entropy'``: ``n_fft`` (int, default ``None``) –
          overrides FFT length (defaults to *window_size*).
        * ``'burst_permutation_entropy'``: ``order`` (int, default ``3``),
          ``delay`` (int, default ``1``).
        * ``'mobility_phase_coherence'``: ``eps`` (float, default ``1e-10``) –
          added to variance before inversion.
    """

    name: str
    ice_component: str
    feature_type: str
    channels: list[str]
    window_size: int = 2
    kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal rolling-window helpers
# ---------------------------------------------------------------------------

def _rolling_apply(
    x: np.ndarray,
    window_size: int,
    func,
    *extra_args,
    **extra_kwargs,
) -> np.ndarray:
    """Apply *func* to every right-aligned rolling window of *x*.

    The first ``window_size − 1`` output samples are set to ``NaN`` (causal
    window).  Output length equals ``len(x)``.

    Parameters
    ----------
    x:
        1-D input array of length *T*.
    window_size:
        Number of samples per window (must be ≥ 1).
    func:
        Callable ``func(window, *extra_args, **extra_kwargs) → float``.
    """
    T = len(x)
    out = np.full(T, np.nan)
    for t in range(window_size - 1, T):
        window = x[t - window_size + 1 : t + 1]
        out[t] = func(window, *extra_args, **extra_kwargs)
    return out


def _rolling_apply_2d(
    X: np.ndarray,
    window_size: int,
    func,
    *extra_args,
    **extra_kwargs,
) -> np.ndarray:
    """Apply *func* to every rolling window of a 2-D array ``(n_ch, T)``.

    Parameters
    ----------
    X:
        2-D input array of shape ``(n_channels, T)``.
    window_size:
        Number of samples per window (must be ≥ 1).
    func:
        Callable ``func(window_2d, *extra_args, **extra_kwargs) → float``
        where ``window_2d`` has shape ``(n_channels, window_size)``.
    """
    n_ch, T = X.shape
    out = np.full(T, np.nan)
    for t in range(window_size - 1, T):
        window = X[:, t - window_size + 1 : t + 1]
        out[t] = func(window, *extra_args, **extra_kwargs)
    return out


# ---------------------------------------------------------------------------
# E – Energy-linked DNA feature functions
# ---------------------------------------------------------------------------

def _rms(window: np.ndarray) -> float:
    """Root-mean-square of *window* (Eq. 24 of the manuscript)."""
    return float(math.sqrt(float(np.mean(window ** 2))))


def compute_fret_energy_proxy(x: np.ndarray, window_size: int) -> np.ndarray:
    """Rolling RMS of a FRET efficiency or intensity time series.

    FRET efficiency fluctuations reflect conformational-energy changes in the
    DNA-associated molecular system.  Rolling RMS provides a time-localised
    energy proxy (Eq. 24):

        E(t) = sqrt( (1/W) Σ_{k=0}^{W-1} x(t−k)² )

    Parameters
    ----------
    x:
        1-D FRET efficiency or donor/acceptor intensity time series.
    window_size:
        Rolling window length in samples (≥ 1).

    Returns
    -------
    np.ndarray of shape ``(T,)`` with NaN for the first ``window_size − 1``
    samples.
    """
    return _rolling_apply(x, window_size, _rms)


def compute_repair_intensity_rms(x: np.ndarray, window_size: int) -> np.ndarray:
    """Rolling RMS of a repair-focus fluorescence intensity trace.

    Repair-focus intensity is coupled to ATP-dependent repair machinery,
    providing an energy proxy for DNA-damage response dynamics (Table 2,
    Energy-linked).  Implements Eq. (24).

    Parameters
    ----------
    x:
        1-D repair-focus intensity time series.
    window_size:
        Rolling window length in samples (≥ 1).

    Returns
    -------
    np.ndarray of shape ``(T,)`` with NaN for the first ``window_size − 1``
    samples.
    """
    return _rolling_apply(x, window_size, _rms)


def compute_replication_stress_rms(x: np.ndarray, window_size: int) -> np.ndarray:
    """Rolling RMS of a replication-stress marker intensity trajectory.

    Stress-marker amplitude reflects energy demands at stalled or active
    replication forks (Table 2, Stress-linked, mapped to E as a fork-activity
    energy proxy).  Implements Eq. (24).

    Parameters
    ----------
    x:
        1-D replication-stress marker intensity time series.
    window_size:
        Rolling window length in samples (≥ 1).

    Returns
    -------
    np.ndarray of shape ``(T,)`` with NaN for the first ``window_size − 1``
    samples.
    """
    return _rolling_apply(x, window_size, _rms)


# ---------------------------------------------------------------------------
# I – Information-linked DNA feature functions
# ---------------------------------------------------------------------------

def _coefficient_of_variation(window: np.ndarray, eps: float) -> float:
    """Coefficient of variation (σ / |μ|) for one window."""
    mu = float(np.mean(window))
    sigma = float(np.std(window))
    return sigma / (abs(mu) + eps)


def compute_burst_variability(
    x: np.ndarray,
    window_size: int,
    *,
    eps: float = 1e-10,
) -> np.ndarray:
    """Rolling coefficient of variation of a transcription burst trace.

    Transcription burst variability (σ / |μ|) quantifies heterogeneity in
    burst amplitude, serving as an Information proxy (Table 2, Information-
    linked: 'transcription burst variability').

    Parameters
    ----------
    x:
        1-D transcription burst intensity time series.
    window_size:
        Rolling window length in samples (≥ 1).
    eps:
        Small constant added to ``|μ|`` to prevent division by zero.

    Returns
    -------
    np.ndarray of shape ``(T,)`` with NaN for the first ``window_size − 1``
    samples.
    """
    return _rolling_apply(x, window_size, _coefficient_of_variation, eps)


def _spectral_entropy(window: np.ndarray, n_fft: int | None = None) -> float:
    """Shannon entropy of the normalised power spectral density (Eq. 25).

    I = −Σ_k p_k log p_k,  where p_k is the normalised one-sided power spectrum.
    """
    n = n_fft if n_fft is not None else len(window)
    fft_mag = np.abs(np.fft.rfft(window, n=n))
    power = fft_mag ** 2
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    p = power / total
    p_nonzero = p[p > 0.0]
    return float(-np.sum(p_nonzero * np.log(p_nonzero)))


def compute_chromatin_entropy(
    x: np.ndarray,
    window_size: int,
    *,
    n_fft: int | None = None,
) -> np.ndarray:
    """Rolling spectral Shannon entropy of a chromatin mobility signal.

    Spectral entropy quantifies conformational or accessibility heterogeneity
    in the chromatin dynamics trace (Table 2, Information-linked: 'chromatin
    accessibility heterogeneity').  Implements the rolling form of Eq. (25):

        I(t) = −Σ_k p_k log p_k

    where p_k is the normalised power spectral density within the window.

    Parameters
    ----------
    x:
        1-D chromatin mobility or accessibility time series.
    window_size:
        Rolling window length in samples (≥ 2 recommended).
    n_fft:
        FFT length (defaults to *window_size*).

    Returns
    -------
    np.ndarray of shape ``(T,)`` with NaN for the first ``window_size − 1``
    samples.
    """
    return _rolling_apply(x, window_size, _spectral_entropy, n_fft)


def _permutation_entropy(window: np.ndarray, order: int, delay: int) -> float:
    """Normalised permutation entropy for one window.

    The probability distribution over the ``order!`` ordinal patterns is used
    to evaluate Shannon entropy, normalised by log(order!) so the result lies
    in [0, 1].
    """
    n = len(window)
    n_patterns = math.factorial(order)
    pattern_count: dict[tuple, int] = {}

    all_perms = list(permutations(range(order)))
    perm_index = {p: i for i, p in enumerate(all_perms)}  # noqa: F841

    n_sub = n - (order - 1) * delay
    if n_sub <= 0:
        return 0.0

    for i in range(n_sub):
        sub = window[i : i + order * delay : delay]
        rank = tuple(int(r) for r in np.argsort(sub, kind="stable"))
        pattern_count[rank] = pattern_count.get(rank, 0) + 1

    total = sum(pattern_count.values())
    if total == 0:
        return 0.0

    probs = np.array([c / total for c in pattern_count.values()])
    h = float(-np.sum(probs * np.log(probs)))
    log_n = math.log(n_patterns) if n_patterns > 1 else 1.0
    return h / log_n


def compute_burst_permutation_entropy(
    x: np.ndarray,
    window_size: int,
    *,
    order: int = 3,
    delay: int = 1,
) -> np.ndarray:
    """Rolling normalised permutation entropy of a transcription burst trace.

    Permutation entropy captures ordinal-pattern complexity and complements
    CV as a burst-variability / information measure (Table 2, Information-
    linked: 'transcription burst variability').  Values lie in ``[0, 1]``.

    Parameters
    ----------
    x:
        1-D transcription burst intensity time series.
    window_size:
        Rolling window length in samples.
    order:
        Ordinal pattern length (embedding dimension), ≥ 2.
    delay:
        Time delay between consecutive elements of each pattern, ≥ 1.

    Returns
    -------
    np.ndarray of shape ``(T,)`` with NaN for the first ``window_size − 1``
    samples.
    """
    return _rolling_apply(x, window_size, _permutation_entropy, order, delay)


# ---------------------------------------------------------------------------
# C – Coherence-linked DNA feature functions
# ---------------------------------------------------------------------------

def _pairwise_synchrony_scalar(window_2d: np.ndarray) -> float:
    """Mean Pearson r across all pairs of channels in *window_2d*.

    *window_2d* has shape ``(n_channels, window_size)``.
    """
    n_ch = window_2d.shape[0]
    if n_ch < 2:
        raise ValueError("pairwise feature requires at least 2 channels.")
    r_values: list[float] = []
    for i, j in combinations(range(n_ch), 2):
        a = window_2d[i]
        b = window_2d[j]
        std_a = float(np.std(a))
        std_b = float(np.std(b))
        if std_a < 1e-12 or std_b < 1e-12:
            r_values.append(0.0)
        else:
            r_values.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(r_values))


def compute_locus_co_motion(
    channels: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Rolling mean pairwise Pearson correlation of locus mobility trajectories.

    Co-motion of genomic loci reflects coherent chromatin reorganisation.
    High pairwise correlation indicates spatially coordinated locus movement
    (Table 2, Coherence-linked: 'locus co-motion').

    Parameters
    ----------
    channels:
        2-D array of shape ``(n_loci, T)`` with ``n_loci ≥ 2``, where each
        row is a locus position or displacement time series.
    window_size:
        Rolling window length in samples (≥ 1).

    Returns
    -------
    np.ndarray of shape ``(T,)`` with NaN for the first ``window_size − 1``
    samples.  Values are in ``[−1, 1]``.
    """
    if channels.ndim != 2 or channels.shape[0] < 2:
        raise ValueError(
            "channels must be a 2-D array of shape (n_loci, T) "
            f"with n_loci ≥ 2; got shape {channels.shape}."
        )
    return _rolling_apply_2d(channels, window_size, _pairwise_synchrony_scalar)


def compute_focus_synchrony(
    channels: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Rolling mean pairwise Pearson correlation of repair-focus intensity channels.

    Synchronous recruitment or dissolution of repair foci indicates coherent
    DNA-damage response dynamics (Table 2, Coherence-linked: 'repair focus
    synchronization').

    Parameters
    ----------
    channels:
        2-D array of shape ``(n_foci, T)`` with ``n_foci ≥ 2``, where each
        row is a repair-focus intensity time series.
    window_size:
        Rolling window length in samples (≥ 1).

    Returns
    -------
    np.ndarray of shape ``(T,)`` with NaN for the first ``window_size − 1``
    samples.  Values are in ``[−1, 1]``.
    """
    if channels.ndim != 2 or channels.shape[0] < 2:
        raise ValueError(
            "channels must be a 2-D array of shape (n_foci, T) "
            f"with n_foci ≥ 2; got shape {channels.shape}."
        )
    return _rolling_apply_2d(channels, window_size, _pairwise_synchrony_scalar)


def _phase_coherence_scalar(window: np.ndarray, eps: float) -> float:
    """Inverse variance of instantaneous phase increments (Eq. 26).

    C = 1 / (Var(Δφ) + ε)

    The instantaneous phase φ(t) is obtained via the analytic signal
    (Hilbert transform).
    """
    analytic = hilbert(window)
    phase = np.unwrap(np.angle(analytic))
    d_phase = np.diff(phase)
    var = float(np.var(d_phase))
    return 1.0 / (var + eps)


def compute_mobility_phase_coherence(
    x: np.ndarray,
    window_size: int,
    *,
    eps: float = 1e-10,
) -> np.ndarray:
    """Rolling phase-coherence proxy for a chromatin locus mobility trace.

    Implements the rolling form of Eq. (26):

        C(t) = 1 / (Var(Δφ(t)) + ε)

    where φ(t) is the instantaneous phase from the Hilbert transform of the
    locus mobility signal.  Stable phase increments indicate coherent locus
    motion.

    Parameters
    ----------
    x:
        1-D chromatin locus mobility (position or displacement) time series.
    window_size:
        Rolling window length in samples (≥ 2 required for phase increments).
    eps:
        Small constant added to variance before inversion.

    Returns
    -------
    np.ndarray of shape ``(T,)`` with NaN for the first ``window_size − 1``
    samples.
    """
    return _rolling_apply(x, window_size, _phase_coherence_scalar, eps)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_dna_spec(
    spec: DNAFeatureSpec,
    data: Mapping[str, np.ndarray],
) -> None:
    """Raise ValueError if *spec* is inconsistent with *data* or the catalogue."""
    if spec.ice_component not in DNA_OBSERVABLE_CATALOGUE:
        raise ValueError(
            f"DNAFeatureSpec '{spec.name}': ice_component must be one of "
            f"{sorted(DNA_OBSERVABLE_CATALOGUE)}; got '{spec.ice_component}'."
        )
    if spec.feature_type not in DNA_OBSERVABLE_CATALOGUE[spec.ice_component]:
        valid = sorted(DNA_OBSERVABLE_CATALOGUE[spec.ice_component])
        raise ValueError(
            f"DNAFeatureSpec '{spec.name}': feature_type '{spec.feature_type}' "
            f"is not valid for ice_component '{spec.ice_component}'. "
            f"Valid types: {valid}."
        )
    for ch in spec.channels:
        if ch not in data:
            raise ValueError(
                f"DNAFeatureSpec '{spec.name}': channel '{ch}' not found in "
                f"data.  Available channels: {sorted(data)}."
            )
    # Channel-count requirements
    ft = spec.feature_type
    n_ch = len(spec.channels)
    if ft in _MULTI_CHANNEL_TYPES:
        if n_ch < 2:
            raise ValueError(
                f"DNAFeatureSpec '{spec.name}': '{ft}' requires ≥ 2 channels; "
                f"got {n_ch}."
            )
    else:
        if n_ch != 1:
            raise ValueError(
                f"DNAFeatureSpec '{spec.name}': feature_type '{ft}' requires "
                f"exactly 1 channel; got {n_ch}."
            )
    # Window size
    if spec.window_size < 2:
        raise ValueError(
            f"DNAFeatureSpec '{spec.name}': window_size must be ≥ 2; "
            f"got {spec.window_size}."
        )


# ---------------------------------------------------------------------------
# map_dna_to_ice
# ---------------------------------------------------------------------------

def map_dna_to_ice(
    data: Mapping[str, np.ndarray],
    feature_specs: Sequence[DNAFeatureSpec],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Compute ICE features from raw DNA-associated time-resolved observables.

    This function constitutes the observable → ICE component mapping described
    in Table 2 and Section 7 of the manuscript.  The caller declares all
    mappings explicitly via :class:`DNAFeatureSpec` objects; no implicit
    assignments are made.

    Allowed input modalities (time-resolved signals only; no static DNA
    sequence interpretation):

    * chromatin mobility trajectories  → ``'locus_co_motion'`` (C),
      ``'chromatin_entropy'`` (I), ``'mobility_phase_coherence'`` (C)
    * FRET time series                 → ``'fret_energy_proxy'`` (E)
    * transcription burst traces       → ``'burst_variability'`` (I),
      ``'burst_permutation_entropy'`` (I)
    * repair-focus dynamics            → ``'repair_intensity_rms'`` (E),
      ``'focus_synchrony'`` (C)
    * replication-stress trajectories  → ``'replication_stress_rms'`` (E)

    Parameters
    ----------
    data:
        Dictionary of raw time-series channels.  Each value must be a 1-D
        NumPy array of the same length *T*.  Example::

            {
                "fret_eff":      fret_array,        # shape (T,)
                "burst_trace":   burst_array,
                "locus_x1":      locus1_array,
                "locus_x2":      locus2_array,
            }

    feature_specs:
        Sequence of :class:`DNAFeatureSpec` objects.  Every ICE component
        (``"E"``, ``"I"``, ``"C"``) must be covered by at least one spec.
        Feature names within the list must be unique.

    Returns
    -------
    observables : dict[str, np.ndarray]
        Computed feature time series keyed by ``spec.name``.  Each array has
        shape ``(T,)`` with NaN for the leading ``window_size − 1`` samples.
        Suitable for passing to :func:`normalize_dna`.

    mapping : dict[str, str]
        ICE component assignment for each feature name.  Suitable for passing
        to :func:`normalize_dna`.

    Raises
    ------
    ValueError
        If any :class:`DNAFeatureSpec` is invalid, any channel is missing,
        not all three ICE components are covered, or feature names are
        duplicated.
    TypeError
        If *data* or *feature_specs* have unexpected types.
    """
    # ------------------------------------------------------------------ #
    # 1. Validate inputs                                                   #
    # ------------------------------------------------------------------ #
    if not isinstance(data, Mapping):
        raise TypeError(
            f"data must be a dict-like object; got {type(data).__name__}."
        )

    arrays: dict[str, np.ndarray] = {}
    ref_length: int | None = None
    for ch_name, arr in data.items():
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 1:
            raise ValueError(
                f"Channel '{ch_name}' must be a 1-D array; got shape {arr.shape}."
            )
        if ref_length is None:
            ref_length = len(arr)
        elif len(arr) != ref_length:
            raise ValueError(
                f"All channels must have the same length; channel '{ch_name}' "
                f"has length {len(arr)}, expected {ref_length}."
            )
        arrays[ch_name] = arr

    if ref_length is None:
        raise ValueError("data must not be empty.")

    seen_names: set[str] = set()
    for spec in feature_specs:
        if spec.name in seen_names:
            raise ValueError(
                f"Duplicate feature name '{spec.name}' in feature_specs."
            )
        seen_names.add(spec.name)
        _validate_dna_spec(spec, arrays)

    covered = {spec.ice_component for spec in feature_specs}
    missing = {"E", "I", "C"} - covered
    if missing:
        raise ValueError(
            f"feature_specs must cover all three ICE components (E, I, C); "
            f"missing: {sorted(missing)}."
        )

    # ------------------------------------------------------------------ #
    # 2. Compute features                                                  #
    # ------------------------------------------------------------------ #
    observables: dict[str, np.ndarray] = {}
    ice_mapping: dict[str, str] = {}

    for spec in feature_specs:
        ft = spec.feature_type
        kw = spec.kwargs

        if ft == "fret_energy_proxy":
            result = compute_fret_energy_proxy(arrays[spec.channels[0]], spec.window_size)

        elif ft == "repair_intensity_rms":
            result = compute_repair_intensity_rms(arrays[spec.channels[0]], spec.window_size)

        elif ft == "replication_stress_rms":
            result = compute_replication_stress_rms(arrays[spec.channels[0]], spec.window_size)

        elif ft == "burst_variability":
            eps = float(kw.get("eps", 1e-10))
            result = compute_burst_variability(
                arrays[spec.channels[0]], spec.window_size, eps=eps
            )

        elif ft == "chromatin_entropy":
            n_fft = kw.get("n_fft", None)
            result = compute_chromatin_entropy(
                arrays[spec.channels[0]], spec.window_size, n_fft=n_fft
            )

        elif ft == "burst_permutation_entropy":
            order = int(kw.get("order", 3))
            delay = int(kw.get("delay", 1))
            result = compute_burst_permutation_entropy(
                arrays[spec.channels[0]], spec.window_size, order=order, delay=delay
            )

        elif ft == "locus_co_motion":
            channels_2d = np.stack(
                [arrays[ch_name] for ch_name in spec.channels], axis=0
            )
            result = compute_locus_co_motion(channels_2d, spec.window_size)

        elif ft == "focus_synchrony":
            channels_2d = np.stack(
                [arrays[ch_name] for ch_name in spec.channels], axis=0
            )
            result = compute_focus_synchrony(channels_2d, spec.window_size)

        elif ft == "mobility_phase_coherence":
            eps = float(kw.get("eps", 1e-10))
            result = compute_mobility_phase_coherence(
                arrays[spec.channels[0]], spec.window_size, eps=eps
            )

        else:
            # Unreachable after _validate_dna_spec; kept for defensive completeness.
            raise ValueError(
                f"DNAFeatureSpec '{spec.name}': unsupported feature_type '{ft}'."
            )

        observables[spec.name] = result
        ice_mapping[spec.name] = spec.ice_component

    return observables, ice_mapping


# ---------------------------------------------------------------------------
# normalize_dna
# ---------------------------------------------------------------------------

def normalize_dna(
    observables: Mapping[str, np.ndarray],
    mapping: Mapping[str, str],
    baseline_window: tuple,
    *,
    min_std: float = 1e-8,
) -> ICEStateSeries:
    """Z-score normalise DNA-associated ICE features and return an ICEStateSeries.

    Applies the z-score normalization described in Section 7 / Eq. (19–20) of
    the manuscript:

        X̃_k(t) = (X_k(t) − μ_k) / σ_k          (k ∈ {E, I, C})

    where μ_k and σ_k are the mean and standard deviation of X_k over the
    reference *baseline_window*.  After normalization the reference state
    corresponds to x* = (0, 0, 0) in the reduced ICE space.

    NaN values in the leading ``window_size − 1`` samples (produced by the
    rolling windows in :func:`map_dna_to_ice`) propagate transparently through
    the normalization arithmetic.

    Parameters
    ----------
    observables:
        Dictionary of feature time series (shape ``(T,)`` each) keyed by
        feature name.  Typically the first return value of
        :func:`map_dna_to_ice`.
    mapping:
        Assignment of feature names to ICE components.  Each key must be
        present in *observables*; each value must be ``"E"``, ``"I"``, or
        ``"C"``.  All three components must be covered.
    baseline_window:
        ``(start, stop)`` index pair (half-open ``[start, stop)``) defining
        the reference stable period.  At least 2 samples must be present.
        NaN values within the window are excluded from mean/std computation.
    min_std:
        Lower bound applied to the computed standard deviation before
        division, preventing division by zero or near-zero values.

    Returns
    -------
    ICEStateSeries
        Normalized deviation series ``(ΔE, ΔI, ΔC)`` of length *T*.

    Raises
    ------
    ValueError
        If *mapping* does not cover all three ICE components, references an
        unknown observable, or if *baseline_window* is invalid.
    TypeError
        If *mapping* is not a dict-like object.
    """
    _REQUIRED = {"E", "I", "C"}

    if not isinstance(mapping, Mapping):
        raise TypeError(
            f"mapping must be a dict-like object; got {type(mapping).__name__}."
        )

    assigned = set(mapping.values())
    missing = _REQUIRED - assigned
    if missing:
        raise ValueError(
            f"mapping must assign all three ICE components (E, I, C); "
            f"missing: {sorted(missing)}."
        )
    extra = assigned - _REQUIRED
    if extra:
        raise ValueError(
            f"mapping contains unknown ICE component(s): {sorted(extra)}. "
            f"Allowed values are 'E', 'I', 'C'."
        )

    if not observables:
        raise ValueError("observables must not be empty.")

    arrays: dict[str, np.ndarray] = {}
    ref_length: int | None = None
    for name, arr in observables.items():
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 1:
            raise ValueError(
                f"Observable '{name}' must be a 1-D array; got shape {arr.shape}."
            )
        if ref_length is None:
            ref_length = len(arr)
        elif len(arr) != ref_length:
            raise ValueError(
                f"All observables must have the same length; "
                f"'{name}' has length {len(arr)}, expected {ref_length}."
            )
        arrays[name] = arr

    for obs_name in mapping:
        if obs_name not in arrays:
            raise ValueError(
                f"mapping references observable '{obs_name}', "
                f"which is not present in observables."
            )

    T = ref_length
    start, stop = int(baseline_window[0]), int(baseline_window[1])
    if not (0 <= start < stop <= T):
        raise ValueError(
            f"baseline_window ({start}, {stop}) is invalid for observables of "
            f"length {T}.  Requires 0 <= start < stop <= T."
        )
    if stop - start < 2:
        raise ValueError(
            f"baseline_window must contain at least 2 samples; "
            f"got {stop - start}."
        )

    # Invert mapping: component → observable name
    comp_to_obs: dict[str, str] = {comp: obs for obs, comp in mapping.items()}

    normalized: dict[str, np.ndarray] = {}
    for comp in ("E", "I", "C"):
        x = arrays[comp_to_obs[comp]]
        baseline_slice = x[start:stop]

        # Use nanmean/nanstd so NaN leading samples (from rolling windows that
        # may overlap the start of the baseline) are excluded transparently.
        mu = float(np.nanmean(baseline_slice))
        sigma = float(np.nanstd(baseline_slice))
        sigma = max(sigma, min_std)

        normalized[comp] = (x - mu) / sigma

    return ICEStateSeries(
        delta_E=normalized["E"],
        delta_I=normalized["I"],
        delta_C=normalized["C"],
    )


# ---------------------------------------------------------------------------
# Convenience end-to-end function
# ---------------------------------------------------------------------------

def dna_feature_mapping(
    data: Mapping[str, np.ndarray],
    feature_specs: Sequence[DNAFeatureSpec],
    baseline_window: tuple,
    *,
    min_std: float = 1e-8,
) -> ICEStateSeries:
    """Map DNA-associated observables to an ICEStateSeries in one call.

    This is the primary entry point for the Stage 7 DNA extension.  It
    combines :func:`map_dna_to_ice` and :func:`normalize_dna` into a single
    convenience function.

    Allowed input modalities (time-resolved signals only; **no static DNA
    sequence interpretation**):

    * **chromatin mobility trajectories** – locus position or displacement
      time series; map to ``'locus_co_motion'`` (C),
      ``'chromatin_entropy'`` (I), or ``'mobility_phase_coherence'`` (C).
    * **FRET time series** – FRET efficiency or donor/acceptor intensity;
      map to ``'fret_energy_proxy'`` (E).
    * **transcription bursts** – single-cell RNA or reporter intensity
      traces; map to ``'burst_variability'`` (I) or
      ``'burst_permutation_entropy'`` (I).
    * **repair-focus dynamics** – repair-focus fluorescence intensity;
      map to ``'repair_intensity_rms'`` (E) or ``'focus_synchrony'`` (C).
    * **replication-stress trajectories** – stress-marker intensity vs.
      time; map to ``'replication_stress_rms'`` (E).

    Parameters
    ----------
    data:
        Dictionary of raw time-series channels.  Each value must be a 1-D
        NumPy array of the same length *T*.
    feature_specs:
        Sequence of :class:`DNAFeatureSpec` objects declaring the feature
        extraction and ICE component assignment.  All three ICE components
        (``"E"``, ``"I"``, ``"C"``) must be covered.
    baseline_window:
        ``(start, stop)`` index pair defining the reference stable period
        used for z-score normalization.  Must satisfy
        ``0 <= start < stop <= T`` and contain at least 2 samples.
    min_std:
        Lower bound on standard deviation used in z-score normalization.

    Returns
    -------
    ICEStateSeries
        Time-resolved normalized ICE deviation series
        ``(ΔE(t), ΔI(t), ΔC(t))`` of length *T*, as defined in
        Eq. (19–20) of the manuscript.

    Raises
    ------
    ValueError
        If inputs are inconsistent (see :func:`map_dna_to_ice` and
        :func:`normalize_dna` for details).

    Examples
    --------
    Minimal three-modality example::

        import numpy as np
        from src.extensions.stage7_dna.dna_feature_mapping import (
            DNAFeatureSpec,
            dna_feature_mapping,
        )

        rng = np.random.default_rng(0)
        T = 500
        data = {
            "fret_eff":  rng.uniform(0.2, 0.8, T),
            "burst":     rng.exponential(1.0, T),
            "locus_a":   np.cumsum(rng.normal(0, 0.1, T)),
            "locus_b":   np.cumsum(rng.normal(0, 0.1, T)),
        }
        specs = [
            DNAFeatureSpec("fret_rms",   "E", "fret_energy_proxy",
                           ["fret_eff"],           window_size=50),
            DNAFeatureSpec("burst_cv",   "I", "burst_variability",
                           ["burst"],              window_size=50),
            DNAFeatureSpec("co_motion",  "C", "locus_co_motion",
                           ["locus_a", "locus_b"], window_size=50),
        ]
        ice = dna_feature_mapping(data, specs, baseline_window=(50, 200))
        # ice.delta_E, ice.delta_I, ice.delta_C – shape (T,) normalized arrays
    """
    observables, mapping = map_dna_to_ice(data, feature_specs)
    return normalize_dna(observables, mapping, baseline_window, min_std=min_std)
