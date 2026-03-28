"""
map_to_ice – Observable-to-ICE component mapping for cellular dynamics.

Implements the explicit, reproducible mapping from cellular observables to the
three triadic ICE components (Energy, Information, Coherence) as described in
the manuscript (Section 5, Table 1, and Section 8.4).

Mapping rules (manuscript Table 1 and Section 3):

  E – Energy / metabolic proxies
      Representative cellular observables: ATP/ADP ratio, mitochondrial
      membrane potential, NADH/FAD redox dynamics.
      Supported feature_type values: 'rms_energy', 'ratio_proxy',
      'direct_proxy'.

  I – Information / entropy–heterogeneity metrics
      Representative cellular observables: transcription entropy,
      gene-regulatory diversity, trajectory complexity.
      Supported feature_type values: 'spectral_entropy',
      'permutation_entropy', 'coefficient_of_variation'.

  C – Coherence / synchrony metrics
      Representative cellular observables: calcium synchrony,
      cell migration alignment, morphodynamic coordination.
      Supported feature_type values: 'phase_coherence',
      'pairwise_synchrony', 'direct_proxy'.

The mapping between measured observables and E / I / C is not unique.
Different experimental systems may require different proxy variables.  The
caller must declare every assignment explicitly via :class:`FeatureSpec`
objects; no implicit or default assignments are made.  The validity of any
particular mapping must be evaluated empirically by the researcher.

No biological interpretation is assigned beyond the observable → ICE component
mapping specified by the caller.

Usage example
-------------
::

    import numpy as np
    from src.extensions.stage6_cellular.map_to_ice import FeatureSpec, map_to_ice

    rng = np.random.default_rng(0)
    T = 500
    data = {
        "atp":      rng.uniform(0.8, 1.2, T),
        "adp":      rng.uniform(0.4, 0.6, T),
        "rna_vals": rng.standard_normal(T),
        "ch1":      np.sin(2 * np.pi * np.arange(T) / 50) + rng.normal(0, 0.1, T),
        "ch2":      np.sin(2 * np.pi * np.arange(T) / 50) + rng.normal(0, 0.1, T),
    }

    specs = [
        FeatureSpec("atp_adp_ratio",  "E", "ratio_proxy",           ["atp", "adp"],   window_size=50),
        FeatureSpec("spectral_H",     "I", "spectral_entropy",       ["rna_vals"],     window_size=50),
        FeatureSpec("pairwise_r",     "C", "pairwise_synchrony",     ["ch1", "ch2"],   window_size=50),
    ]

    observables, mapping = map_to_ice(data, specs)
    # Pass to normalize_cellular:
    # from src.extensions.stage6_cellular.normalize_cellular import normalize_cellular
    # ice = normalize_cellular(observables, mapping, baseline_window=(0, 100))
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

# Ensure the repository root is on sys.path so that src.core imports work
# regardless of the working directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Observable catalogue
# ---------------------------------------------------------------------------

#: Maps ICE component → {feature_type: short description}.
#: This catalogue is the authoritative record of supported observables and
#: their component assignments.  It is derived directly from Table 1 and
#: Section 8.4 of the manuscript.
OBSERVABLE_CATALOGUE: dict[str, dict[str, str]] = {
    "E": {
        "rms_energy": (
            "Rolling root-mean-square (Eq. 24).  Suitable for amplitude- or "
            "intensity-based energy proxies such as mitochondrial potential "
            "or NADH/FAD fluorescence."
        ),
        "ratio_proxy": (
            "Element-wise ratio of two channels (numerator / denominator).  "
            "Suitable for metabolic ratios such as ATP/ADP or NAD+/NADH."
        ),
        "direct_proxy": (
            "Direct passthrough of a single pre-computed observable.  Use "
            "when the caller has already extracted a suitable energy proxy."
        ),
    },
    "I": {
        "spectral_entropy": (
            "Rolling spectral Shannon entropy (Eq. 25): H = −Σ p_k log p_k "
            "where p_k is the normalised power spectral density.  Suitable "
            "for trajectory complexity and transcriptional heterogeneity."
        ),
        "permutation_entropy": (
            "Rolling permutation entropy (ordinal patterns).  Suitable for "
            "symbolic complexity and burst heterogeneity measures."
        ),
        "coefficient_of_variation": (
            "Rolling coefficient of variation (σ / |μ|).  Suitable for "
            "gene-regulatory diversity and state-space dispersion metrics."
        ),
    },
    "C": {
        "phase_coherence": (
            "Inverse variance of instantaneous phase increments (Eq. 26): "
            "C = 1 / (Var(Δφ) + ε).  Suitable for calcium synchrony and "
            "oscillatory coordination."
        ),
        "pairwise_synchrony": (
            "Mean pairwise Pearson correlation across ≥ 2 channels within "
            "each rolling window.  Suitable for multichannel synchrony such "
            "as calcium imaging or migration alignment."
        ),
        "direct_proxy": (
            "Direct passthrough of a single pre-computed observable.  Use "
            "when the caller has already extracted a suitable coherence proxy."
        ),
    },
}

# Flat set of all valid (feature_type, ice_component) pairs.
_VALID_ASSIGNMENTS: frozenset[tuple[str, str]] = frozenset(
    (ft, comp)
    for comp, ftypes in OBSERVABLE_CATALOGUE.items()
    for ft in ftypes
)


# ---------------------------------------------------------------------------
# FeatureSpec
# ---------------------------------------------------------------------------

@dataclass
class FeatureSpec:
    """Declare one feature to be extracted and its ICE component assignment.

    Parameters
    ----------
    name:
        Unique output name for the computed feature time series.  This name
        will appear as a key in the *observables* dict returned by
        :func:`map_to_ice`.

    ice_component:
        ICE component this feature is assigned to.  Must be one of
        ``"E"``, ``"I"``, or ``"C"``.

    feature_type:
        Algorithm used to extract the feature.  Must be one of the keys
        listed in :data:`OBSERVABLE_CATALOGUE` for the given
        *ice_component*.

    channels:
        Names of the raw-data channels (keys in the *data* dict passed to
        :func:`map_to_ice`) required by this feature.

        - ``'rms_energy'``, ``'spectral_entropy'``, ``'permutation_entropy'``,
          ``'coefficient_of_variation'``, ``'phase_coherence'``,
          ``'direct_proxy'``: exactly one channel.
        - ``'ratio_proxy'``: exactly two channels ``[numerator, denominator]``.
        - ``'pairwise_synchrony'``: two or more channels.

    window_size:
        Rolling window length in samples used by windowed feature types.
        Ignored for ``'direct_proxy'``.  Must be ≥ 2.

    kwargs:
        Additional keyword arguments forwarded to the underlying feature
        function.  Supported overrides:

        - ``'ratio_proxy'``: ``eps`` (float, default ``1e-10``) – added to
          denominator before division.
        - ``'spectral_entropy'``: ``n_fft`` (int, default ``None``) –
          overrides the FFT length (defaults to *window_size*).
        - ``'permutation_entropy'``: ``order`` (int, default ``3``),
          ``delay`` (int, default ``1``).
        - ``'coefficient_of_variation'``: ``eps`` (float, default ``1e-10``) –
          added to |μ| before division.
        - ``'phase_coherence'``: ``eps`` (float, default ``1e-10``) – added
          to variance before inversion.
    """

    name: str
    ice_component: str
    feature_type: str
    channels: list[str]
    window_size: int = 1
    kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal rolling-window helper
# ---------------------------------------------------------------------------

def _rolling_apply(
    x: np.ndarray,
    window_size: int,
    func,
    *extra_args,
    **extra_kwargs,
) -> np.ndarray:
    """Apply *func* to every rolling window of length *window_size* in *x*.

    The first ``window_size − 1`` output samples are set to ``NaN`` (causal,
    right-aligned window).  Output length equals ``len(x)``.

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
# E – Energy / metabolic proxy features
# ---------------------------------------------------------------------------

def _rms_energy(window: np.ndarray) -> float:
    """Root-mean-square of *window* (Eq. 24 of the manuscript)."""
    return float(math.sqrt(float(np.mean(window ** 2))))


def compute_rms_energy(x: np.ndarray, window_size: int) -> np.ndarray:
    """Rolling RMS energy proxy.

    Implements Eq. (24):  E(t) = sqrt( (1/W) Σ_{k} s_k² )

    Parameters
    ----------
    x:
        1-D time series.
    window_size:
        Rolling window length in samples (≥ 1).

    Returns
    -------
    np.ndarray of shape ``(T,)`` with NaN for the first ``window_size − 1``
    samples.
    """
    return _rolling_apply(x, window_size, _rms_energy)


def compute_ratio_proxy(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    eps: float = 1e-10,
) -> np.ndarray:
    """Element-wise ratio proxy (e.g., ATP/ADP).

    No rolling window is applied; the ratio is computed point-wise.

    Parameters
    ----------
    numerator, denominator:
        1-D arrays of the same length *T*.
    eps:
        Small constant added to *denominator* to prevent division by zero.

    Returns
    -------
    np.ndarray of shape ``(T,)``.
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    return num / (den + eps)


# ---------------------------------------------------------------------------
# I – Information / entropy–heterogeneity features
# ---------------------------------------------------------------------------

def _spectral_entropy(window: np.ndarray, n_fft: int | None = None) -> float:
    """Shannon entropy of the normalised power spectral density.

    Implements Eq. (25):  I = −Σ_k p_k log p_k
    where p_k is the one-sided normalised power spectrum.
    """
    n = n_fft if n_fft is not None else len(window)
    fft_mag = np.abs(np.fft.rfft(window, n=n))
    power = fft_mag ** 2
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    p = power / total
    # Replace zeros before log to avoid -inf; treat 0 log 0 = 0.
    p_nonzero = p[p > 0.0]
    return float(-np.sum(p_nonzero * np.log(p_nonzero)))


def compute_spectral_entropy(
    x: np.ndarray,
    window_size: int,
    *,
    n_fft: int | None = None,
) -> np.ndarray:
    """Rolling spectral Shannon entropy.

    Implements the rolling form of Eq. (25).

    Parameters
    ----------
    x:
        1-D time series.
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


def _permutation_entropy(
    window: np.ndarray,
    order: int,
    delay: int,
) -> float:
    """Normalised permutation entropy for one window.

    For each consecutive subsequence of length *order* (spaced by *delay*)
    the ordinal pattern rank is computed.  The probability distribution over
    the ``order!`` possible patterns is used to evaluate Shannon entropy,
    normalised by log(order!) so the result lies in [0, 1].
    """
    n = len(window)
    n_patterns = math.factorial(order)
    pattern_count: dict[tuple, int] = {}

    # Build ordinal-pattern index table
    all_perms = list(permutations(range(order)))
    perm_index = {p: i for i, p in enumerate(all_perms)}

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


def compute_permutation_entropy(
    x: np.ndarray,
    window_size: int,
    *,
    order: int = 3,
    delay: int = 1,
) -> np.ndarray:
    """Rolling normalised permutation entropy.

    Parameters
    ----------
    x:
        1-D time series.
    window_size:
        Rolling window length in samples.
    order:
        Ordinal pattern length (embedding dimension), ≥ 2.
    delay:
        Time delay between consecutive elements of each pattern, ≥ 1.

    Returns
    -------
    np.ndarray of shape ``(T,)`` with NaN for the first ``window_size − 1``
    samples.  Values are in ``[0, 1]`` (normalised by log(order!)).
    """
    return _rolling_apply(x, window_size, _permutation_entropy, order, delay)


def _coefficient_of_variation(window: np.ndarray, eps: float) -> float:
    """Coefficient of variation (σ / |μ|) for one window."""
    mu = float(np.mean(window))
    sigma = float(np.std(window))
    return sigma / (abs(mu) + eps)


def compute_coefficient_of_variation(
    x: np.ndarray,
    window_size: int,
    *,
    eps: float = 1e-10,
) -> np.ndarray:
    """Rolling coefficient of variation (σ / |μ|).

    Used as a heterogeneity / gene-regulatory-diversity proxy.

    Parameters
    ----------
    x:
        1-D time series.
    window_size:
        Rolling window length in samples (≥ 1).
    eps:
        Small constant added to |μ| to prevent division by zero.

    Returns
    -------
    np.ndarray of shape ``(T,)`` with NaN for the first ``window_size − 1``
    samples.
    """
    return _rolling_apply(x, window_size, _coefficient_of_variation, eps)


# ---------------------------------------------------------------------------
# C – Coherence / synchrony features
# ---------------------------------------------------------------------------

def _phase_coherence_scalar(window: np.ndarray, eps: float) -> float:
    """Inverse variance of instantaneous phase increments (Eq. 26).

    C = 1 / (Var(Δφ) + ε)

    The instantaneous phase φ(t) is obtained via the analytic signal
    (Hilbert transform).  Phase increments Δφ = diff(unwrap(φ)) are used.
    """
    analytic = hilbert(window)
    phase = np.unwrap(np.angle(analytic))
    d_phase = np.diff(phase)
    var = float(np.var(d_phase))
    return 1.0 / (var + eps)


def compute_phase_coherence(
    x: np.ndarray,
    window_size: int,
    *,
    eps: float = 1e-10,
) -> np.ndarray:
    """Rolling phase-coherence proxy.

    Implements the rolling form of Eq. (26):
    C(t) = 1 / (Var(Δφ(t)) + ε)

    where φ(t) is the instantaneous phase obtained via the Hilbert transform.

    Parameters
    ----------
    x:
        1-D time series.
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


def _pairwise_synchrony_scalar(window_2d: np.ndarray) -> float:
    """Mean Pearson r across all pairs of channels in *window_2d*.

    *window_2d* has shape ``(n_channels, window_size)``.
    """
    n_ch = window_2d.shape[0]
    if n_ch < 2:
        raise ValueError("pairwise_synchrony requires at least 2 channels.")
    r_values: list[float] = []
    for i, j in combinations(range(n_ch), 2):
        a = window_2d[i]
        b = window_2d[j]
        # Pearson r; guard against constant windows.
        std_a = float(np.std(a))
        std_b = float(np.std(b))
        if std_a < 1e-12 or std_b < 1e-12:
            r_values.append(0.0)
        else:
            r_values.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(r_values))


def compute_pairwise_synchrony(
    channels: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Rolling mean pairwise Pearson correlation across channels.

    Parameters
    ----------
    channels:
        2-D array of shape ``(n_channels, T)`` with ``n_channels ≥ 2``.
    window_size:
        Rolling window length in samples (≥ 1).

    Returns
    -------
    np.ndarray of shape ``(T,)`` with NaN for the first ``window_size − 1``
    samples.  Values are in ``[−1, 1]``.
    """
    if channels.ndim != 2 or channels.shape[0] < 2:
        raise ValueError(
            "channels must be a 2-D array of shape (n_channels, T) "
            f"with n_channels ≥ 2; got shape {channels.shape}."
        )
    return _rolling_apply_2d(channels, window_size, _pairwise_synchrony_scalar)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_spec(spec: FeatureSpec, data: Mapping[str, np.ndarray]) -> None:
    """Raise ValueError if *spec* is inconsistent."""
    if spec.ice_component not in OBSERVABLE_CATALOGUE:
        raise ValueError(
            f"FeatureSpec '{spec.name}': ice_component must be one of "
            f"{sorted(OBSERVABLE_CATALOGUE)}; got '{spec.ice_component}'."
        )
    if spec.feature_type not in OBSERVABLE_CATALOGUE[spec.ice_component]:
        valid = sorted(OBSERVABLE_CATALOGUE[spec.ice_component])
        raise ValueError(
            f"FeatureSpec '{spec.name}': feature_type '{spec.feature_type}' "
            f"is not valid for ice_component '{spec.ice_component}'. "
            f"Valid types: {valid}."
        )
    for ch in spec.channels:
        if ch not in data:
            raise ValueError(
                f"FeatureSpec '{spec.name}': channel '{ch}' not found in data. "
                f"Available channels: {sorted(data)}."
            )
    # Channel-count requirements
    ft = spec.feature_type
    n_ch = len(spec.channels)
    if ft == "ratio_proxy":
        if n_ch != 2:
            raise ValueError(
                f"FeatureSpec '{spec.name}': 'ratio_proxy' requires exactly "
                f"2 channels [numerator, denominator]; got {n_ch}."
            )
    elif ft == "pairwise_synchrony":
        if n_ch < 2:
            raise ValueError(
                f"FeatureSpec '{spec.name}': 'pairwise_synchrony' requires "
                f"≥ 2 channels; got {n_ch}."
            )
    else:
        if n_ch != 1:
            raise ValueError(
                f"FeatureSpec '{spec.name}': feature_type '{ft}' requires "
                f"exactly 1 channel; got {n_ch}."
            )
    # Window size
    if ft not in {"ratio_proxy", "direct_proxy"}:
        if spec.window_size < 2:
            raise ValueError(
                f"FeatureSpec '{spec.name}': window_size must be ≥ 2 for "
                f"feature_type '{ft}'; got {spec.window_size}."
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_to_ice(
    data: Mapping[str, np.ndarray],
    feature_specs: Sequence[FeatureSpec],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Compute ICE features from raw cellular observables.

    This function constitutes the observable → ICE component mapping described
    in Table 1 and Section 8.4 of the manuscript.  The caller declares all
    mappings explicitly via :class:`FeatureSpec` objects; no implicit
    assignments are made.

    Parameters
    ----------
    data:
        Dictionary of raw channel time series.  Each value must be a 1-D
        NumPy array of the same length *T*.  Example::

            {
                "atp":       atp_array,    # shape (T,)
                "adp":       adp_array,
                "rna_trace": rna_array,
                "ca_ch1":    ca1_array,
                "ca_ch2":    ca2_array,
            }

    feature_specs:
        Sequence of :class:`FeatureSpec` objects, one per derived feature.
        Every ICE component (``"E"``, ``"I"``, ``"C"``) must be covered by at
        least one spec.  Feature names within the list must be unique.

    Returns
    -------
    observables : dict[str, np.ndarray]
        Computed feature time series keyed by ``spec.name``.  Suitable for
        passing directly to :func:`~normalize_cellular.normalize_cellular`.

    mapping : dict[str, str]
        ICE component assignment for each feature name.  When multiple specs
        map to the same ICE component the last one (by list order) is used in
        downstream normalization (the ``normalize_cellular`` contract allows
        only one observable per component).  Suitable for passing directly to
        :func:`~normalize_cellular.normalize_cellular`.

    Raises
    ------
    ValueError
        If any :class:`FeatureSpec` is invalid, any channel is missing, not
        all three ICE components are covered, or feature names are duplicated.
    TypeError
        If *data* or *feature_specs* have unexpected types.

    Notes
    -----
    The returned *observables* and *mapping* are immediately compatible with
    :func:`~normalize_cellular.normalize_cellular`.  When several specs share
    the same ICE component the caller should ensure that exactly one spec per
    component is intended for downstream use, or aggregate the channels first.

    The mapping from observable → ICE component is conservative and explicit:
    no heuristics or defaults are applied.
    """
    # ------------------------------------------------------------------ #
    # 1. Coerce and validate inputs                                        #
    # ------------------------------------------------------------------ #
    if not isinstance(data, Mapping):
        raise TypeError(
            f"data must be a dict-like object; got {type(data).__name__}."
        )

    # Coerce all data arrays and check uniform length.
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

    # Validate specs.
    seen_names: set[str] = set()
    for spec in feature_specs:
        if spec.name in seen_names:
            raise ValueError(
                f"Duplicate feature name '{spec.name}' in feature_specs."
            )
        seen_names.add(spec.name)
        _validate_spec(spec, arrays)

    # Check all three ICE components are covered.
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

        if ft == "rms_energy":
            ch = arrays[spec.channels[0]]
            result = compute_rms_energy(ch, spec.window_size)

        elif ft == "ratio_proxy":
            num = arrays[spec.channels[0]]
            den = arrays[spec.channels[1]]
            eps = float(kw.get("eps", 1e-10))
            result = compute_ratio_proxy(num, den, eps=eps)

        elif ft == "direct_proxy":
            result = arrays[spec.channels[0]].copy()

        elif ft == "spectral_entropy":
            ch = arrays[spec.channels[0]]
            n_fft = kw.get("n_fft", None)
            result = compute_spectral_entropy(ch, spec.window_size, n_fft=n_fft)

        elif ft == "permutation_entropy":
            ch = arrays[spec.channels[0]]
            order = int(kw.get("order", 3))
            delay = int(kw.get("delay", 1))
            result = compute_permutation_entropy(
                ch, spec.window_size, order=order, delay=delay
            )

        elif ft == "coefficient_of_variation":
            ch = arrays[spec.channels[0]]
            eps = float(kw.get("eps", 1e-10))
            result = compute_coefficient_of_variation(ch, spec.window_size, eps=eps)

        elif ft == "phase_coherence":
            ch = arrays[spec.channels[0]]
            eps = float(kw.get("eps", 1e-10))
            result = compute_phase_coherence(ch, spec.window_size, eps=eps)

        elif ft == "pairwise_synchrony":
            channels_2d = np.stack(
                [arrays[ch_name] for ch_name in spec.channels], axis=0
            )  # shape (n_ch, T)
            result = compute_pairwise_synchrony(channels_2d, spec.window_size)

        else:
            # This branch is unreachable after _validate_spec, but kept for
            # defensive completeness.
            raise ValueError(
                f"FeatureSpec '{spec.name}': unsupported feature_type '{ft}'."
            )

        observables[spec.name] = result
        ice_mapping[spec.name] = spec.ice_component

    return observables, ice_mapping
