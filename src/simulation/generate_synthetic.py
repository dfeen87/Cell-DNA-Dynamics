"""
Synthetic ICE time-series generator for the triadic stability framework.

Produces a fully deterministic ``ICEStateSeries`` whose three components
(ΔE, ΔI, ΔC) pass through four canonical regimes:

    [0, t1)   – stable            (low-amplitude, low-noise)
    [t1, t2)  – pre-instability   (drifting amplitude/frequency)
    [t2, t3)  – instability       (high-amplitude, high-noise)
    [t3, T)   – recovery          (returning toward baseline)

The function also writes ``data/synthetic/eic_timeseries.csv``,
``data/synthetic/delta_phi.csv``, and ``data/synthetic/regimes.csv`` with
columns that are compatible with downstream modules.

Usage
-----
>>> from src.simulation.generate_synthetic import generate_synthetic_ice
>>> series, t, regimes = generate_synthetic_ice()
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrapping – allow running from the repo root without installing.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from src.core.models import ICEStateSeries  # noqa: E402


# ---------------------------------------------------------------------------
# Default simulation parameters
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "fs": 50,          # Hz
    "T": 180.0,        # total duration (s)
    "t1": 45.0,        # stable → pre-instability (s)
    "t2": 90.0,        # pre-instability → instability (s)
    "t3": 135.0,       # instability → recovery (s)
    "noise_scale": 0.05,
    "seed": 0,
}


def _build_component(
    t: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
    amp_stable: float,
    amp_pre: float,
    amp_instab: float,
    amp_recovery: float,
    freq: float,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one ICE component with four-regime dynamics."""
    n = len(t)
    signal = np.zeros(n)

    stable_mask = t < t1
    pre_mask = (t >= t1) & (t < t2)
    instab_mask = (t >= t2) & (t < t3)
    recovery_mask = t >= t3

    # Stable – small sinusoidal oscillation around zero
    signal[stable_mask] = (
        amp_stable * np.sin(2 * np.pi * freq * t[stable_mask])
        + noise_scale * rng.standard_normal(stable_mask.sum())
    )

    # Pre-instability – gradually rising amplitude
    tau_pre = (t[pre_mask] - t1) / (t2 - t1)  # 0→1
    signal[pre_mask] = (
        (amp_stable + (amp_pre - amp_stable) * tau_pre)
        * np.sin(2 * np.pi * freq * t[pre_mask])
        + (noise_scale * (1 + tau_pre)) * rng.standard_normal(pre_mask.sum())
    )

    # Instability – large, noisy excursions
    signal[instab_mask] = (
        amp_instab * np.sin(2 * np.pi * freq * t[instab_mask])
        + (noise_scale * 4) * rng.standard_normal(instab_mask.sum())
    )

    # Recovery – decaying back toward zero
    tau_rec = (t[recovery_mask] - t3) / max(t[-1] - t3, 1e-6)  # 0→1
    signal[recovery_mask] = (
        (amp_instab * (1 - tau_rec) + amp_recovery * tau_rec)
        * np.sin(2 * np.pi * freq * t[recovery_mask])
        + (noise_scale * (2 - tau_rec)) * rng.standard_normal(recovery_mask.sum())
    )

    return signal


def _zscore_baseline(arr: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    """Z-score normalise *arr* using the mean/std of the baseline segment."""
    mu = np.mean(arr[baseline_mask])
    sigma = np.std(arr[baseline_mask]) + 1e-8
    return (arr - mu) / sigma


def generate_synthetic_ice(
    fs: float = _DEFAULTS["fs"],
    T: float = _DEFAULTS["T"],
    t1: float = _DEFAULTS["t1"],
    t2: float = _DEFAULTS["t2"],
    t3: float = _DEFAULTS["t3"],
    noise_scale: float = _DEFAULTS["noise_scale"],
    seed: int = _DEFAULTS["seed"],
) -> tuple[ICEStateSeries, np.ndarray, np.ndarray]:
    """Generate a synthetic ``ICEStateSeries`` with four canonical regimes.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    T : float
        Total duration in seconds.
    t1, t2, t3 : float
        Regime-boundary times (s).  Must satisfy 0 < t1 < t2 < t3 < T.
    noise_scale : float
        Base noise amplitude; scaled by regime inside the function.
    seed : int
        Random seed for full reproducibility.

    Returns
    -------
    series : ICEStateSeries
        Z-score-normalised ΔE(t), ΔI(t), ΔC(t) arrays.
    t : np.ndarray, shape (N,)
        Time vector in seconds.
    regime_labels : np.ndarray, shape (N,), dtype object
        Per-sample regime strings drawn from
        ``{"stable", "pre-instability", "instability", "recovery"}``.

    Side-effects
    ------------
    Writes three CSV files under ``data/synthetic/``:

    * ``eic_timeseries.csv``  – columns: time, delta_E, delta_I, delta_C
    * ``delta_phi.csv``       – columns: time, delta_phi
    * ``regimes.csv``         – columns: time, regime
    """
    rng = np.random.default_rng(seed)

    N = int(round(T * fs))
    t = np.arange(N) / fs

    # Build raw (un-normalised) components with distinct dynamics
    raw_E = _build_component(
        t, t1, t2, t3,
        amp_stable=0.3, amp_pre=0.8, amp_instab=2.0, amp_recovery=0.4,
        freq=0.25, noise_scale=noise_scale, rng=rng,
    )
    raw_I = _build_component(
        t, t1, t2, t3,
        amp_stable=0.2, amp_pre=0.6, amp_instab=1.8, amp_recovery=0.35,
        freq=0.30, noise_scale=noise_scale * 0.8, rng=rng,
    )
    raw_C = _build_component(
        t, t1, t2, t3,
        amp_stable=0.25, amp_pre=0.7, amp_instab=1.5, amp_recovery=0.3,
        freq=0.20, noise_scale=noise_scale * 1.2, rng=rng,
    )

    # Z-score normalise against the stable baseline [0, t1)
    baseline_mask = t < t1
    delta_E = _zscore_baseline(raw_E, baseline_mask)
    delta_I = _zscore_baseline(raw_I, baseline_mask)
    delta_C = _zscore_baseline(raw_C, baseline_mask)

    series = ICEStateSeries(delta_E=delta_E, delta_I=delta_I, delta_C=delta_C)

    # Regime labels
    regime_labels = np.empty(N, dtype=object)
    regime_labels[t < t1] = "stable"
    regime_labels[(t >= t1) & (t < t2)] = "pre-instability"
    regime_labels[(t >= t2) & (t < t3)] = "instability"
    regime_labels[t >= t3] = "recovery"

    # ΔΦ(t) = Euclidean norm of the ICE deviation vector
    delta_phi = np.sqrt(delta_E ** 2 + delta_I ** 2 + delta_C ** 2)

    # ------------------------------------------------------------------
    # Persist outputs
    # ------------------------------------------------------------------
    out_dir = os.path.join(_REPO_ROOT, "data", "synthetic")
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame({
        "time": t,
        "delta_E": delta_E,
        "delta_I": delta_I,
        "delta_C": delta_C,
    }).to_csv(os.path.join(out_dir, "eic_timeseries.csv"), index=False)

    pd.DataFrame({
        "time": t,
        "delta_phi": delta_phi,
    }).to_csv(os.path.join(out_dir, "delta_phi.csv"), index=False)

    pd.DataFrame({
        "time": t,
        "regime": regime_labels,
    }).to_csv(os.path.join(out_dir, "regimes.csv"), index=False)

    return series, t, regime_labels


if __name__ == "__main__":
    series, t, regimes = generate_synthetic_ice()
    print(f"Generated ICEStateSeries with T={len(t)} samples.")
    print(f"Regimes: {dict(zip(*np.unique(regimes, return_counts=True)))}")
    print("Saved: data/synthetic/eic_timeseries.csv, delta_phi.csv, regimes.csv")
