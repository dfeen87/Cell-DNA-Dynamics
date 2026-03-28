"""
normalize_cellular – ICE-consistent normalization for cellular observables.

The ICE (Energy–Information–Coherence) framework requires that user-supplied
time-series observables be explicitly assigned to the three triadic components
E, I, and C before analysis.  This module enforces that explicit mapping and
applies z-score normalization against a caller-defined baseline window, as
described in the manuscript (Section 5):

    X̃_k(t) = (X_k(t) − μ_k) / σ_k          (k ∈ {E, I, C})

where μ_k and σ_k are the mean and standard deviation of X_k over the
reference baseline period.  After normalization the reference state
corresponds to x* = (0, 0, 0) in the reduced ICE space.

No biological interpretation is assigned to the observables beyond the
caller-specified E / I / C mapping.  The validity of any particular mapping
must be evaluated empirically by the researcher.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import numpy as np

# Ensure the repository root is on sys.path so that `src.core.models` can be
# imported regardless of the working directory.  This is necessary because the
# project does not yet define an installable package (no setup.py /
# pyproject.toml / __init__.py files are present).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.models import ICEStateSeries  # noqa: E402


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_cellular(
    observables: Mapping[str, np.ndarray],
    mapping: Mapping[str, str],
    baseline_window: tuple,
    *,
    min_std: float = 1e-8,
) -> ICEStateSeries:
    """Normalize cellular observables and return an :class:`ICEStateSeries`.

    Parameters
    ----------
    observables:
        Dictionary whose keys are observable names and whose values are 1-D
        NumPy arrays of the same length *T* (the time-series samples).
        Example::

            {
                "atp_ratio":        atp_array,      # shape (T,)
                "spectral_entropy":  se_array,
                "phase_coherence":   pc_array,
            }

    mapping:
        Explicit assignment of observable names to ICE components.  Each key
        must be an observable name present in *observables*; each value must
        be exactly one of ``"E"``, ``"I"``, or ``"C"``.  All three components
        must be covered.  Example::

            {
                "atp_ratio":        "E",
                "spectral_entropy":  "I",
                "phase_coherence":   "C",
            }

    baseline_window:
        ``(start, stop)`` index pair (half-open, i.e. ``[start, stop)``)
        defining the reference baseline period.  The mean and standard
        deviation used for z-score normalization are computed exclusively over
        this window.  The window must contain at least two samples so that a
        meaningful standard deviation can be estimated.

    min_std:
        Lower bound applied to the computed standard deviation before
        division, preventing division by zero or near-zero values.  Defaults
        to ``1e-8``.

    Returns
    -------
    ICEStateSeries
        Normalized deviation series ``(ΔE, ΔI, ΔC)`` of length *T*.  By
        construction, the baseline-window mean of each component is zero and
        the baseline-window standard deviation is one (up to *min_std*
        clamping).

    Raises
    ------
    ValueError
        If *mapping* does not cover all three ICE components, references an
        unknown observable, or if *baseline_window* is out of range or too
        short.
    TypeError
        If *mapping* is not a dict-like object.
    """
    # ------------------------------------------------------------------ #
    # 1. Validate mapping                                                  #
    # ------------------------------------------------------------------ #
    _REQUIRED_COMPONENTS = {"E", "I", "C"}

    if not isinstance(mapping, Mapping):
        raise TypeError(
            f"mapping must be a dict-like object; got {type(mapping).__name__}."
        )

    assigned_components = set(mapping.values())
    missing = _REQUIRED_COMPONENTS - assigned_components
    if missing:
        raise ValueError(
            f"mapping must assign all three ICE components (E, I, C); "
            f"missing: {sorted(missing)}."
        )

    extra = assigned_components - _REQUIRED_COMPONENTS
    if extra:
        raise ValueError(
            f"mapping contains unknown ICE component(s): {sorted(extra)}. "
            f"Allowed values are 'E', 'I', 'C'."
        )

    # ------------------------------------------------------------------ #
    # 2. Validate observables                                              #
    # ------------------------------------------------------------------ #
    if not observables:
        raise ValueError("observables must not be empty.")

    arrays: dict = {}
    reference_length = None

    for name, arr in observables.items():
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 1:
            raise ValueError(
                f"Observable '{name}' must be a 1-D array; got shape {arr.shape}."
            )
        if reference_length is None:
            reference_length = len(arr)
        elif len(arr) != reference_length:
            raise ValueError(
                f"All observables must have the same length; "
                f"observable '{name}' has length {len(arr)}, expected {reference_length}."
            )
        arrays[name] = arr

    for obs_name in mapping:
        if obs_name not in arrays:
            raise ValueError(
                f"mapping references observable '{obs_name}', "
                f"which is not present in observables."
            )

    T = reference_length

    # ------------------------------------------------------------------ #
    # 3. Validate baseline window                                          #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 4. Build component arrays (E, I, C)                                 #
    # ------------------------------------------------------------------ #
    # Invert mapping: component → observable name
    component_to_obs: dict = {comp: obs for obs, comp in mapping.items()}

    raw: dict = {
        comp: arrays[obs_name]
        for comp, obs_name in component_to_obs.items()
    }

    # ------------------------------------------------------------------ #
    # 5. Z-score normalization over baseline window                        #
    # ------------------------------------------------------------------ #
    normalized: dict = {}

    for comp in ("E", "I", "C"):
        x = raw[comp]
        baseline_slice = x[start:stop]

        mu = float(np.mean(baseline_slice))
        sigma = float(np.std(baseline_slice))
        sigma = max(sigma, min_std)

        normalized[comp] = (x - mu) / sigma

    # ------------------------------------------------------------------ #
    # 6. Return ICEStateSeries                                             #
    # ------------------------------------------------------------------ #
    return ICEStateSeries(
        delta_E=normalized["E"],
        delta_I=normalized["I"],
        delta_C=normalized["C"],
    )
