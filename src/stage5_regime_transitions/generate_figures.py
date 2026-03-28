"""
Stage 5 Figure Generation – Regime-Transition Detection
========================================================

Generates two figures using the Stage 5 operators:

1. figures/stage5_regime_transitions/delta_phi_thresholds.png
   - ΔΦ(t) curve over time
   - Horizontal dashed lines for θ_low and θ_high thresholds

2. figures/stage5_regime_transitions/regime_segmentation.png
   - Broad colored regime bands (stable / pre-instability / instability / recovery)
   - Clean vertical boundary lines at regime transitions
   - ΔΦ(t) overlaid as a single curve

Reproducibility settings:
    seed = 0, DPI = 150, figure sizes as specified per figure below.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ── Allow running from repository root or from this file's directory ──────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from src.core.models import ICEStateSeries
from src.stage5_regime_transitions.compute_delta_phi import compute_delta_phi
from src.stage5_regime_transitions.detect_regimes import detect_regimes
from src.stage5_regime_transitions.segmentation_utils import (
    regime_shading_intervals,
    extract_transition_events,
)

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 0
DPI = 150
np.random.seed(SEED)

# ── Consistent regime colour palette (Stage 2 / Stage 3 compatible) ──────────
REGIME_COLORS = {
    "stable":          "#2196F3",   # blue
    "pre-instability": "#FF9800",   # orange
    "instability":     "#F44336",   # red
    "recovery":        "#4CAF50",   # green
}
REGIME_ALPHA = 0.18

# ── Paths ─────────────────────────────────────────────────────────────────────
_DATA_PATH = os.path.join(_REPO_ROOT, "data", "synthetic", "eic_timeseries.csv")
_FIG_DIR = os.path.join(_REPO_ROOT, "figures", "stage5_regime_transitions")


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_series(path: str) -> tuple[ICEStateSeries, np.ndarray]:
    """Load eic_timeseries.csv and return (ICEStateSeries, time array)."""
    df = pd.read_csv(path)
    t = df["time"].to_numpy(dtype=float)
    series = ICEStateSeries(
        delta_E=df["delta_E"].to_numpy(dtype=float),
        delta_I=df["delta_I"].to_numpy(dtype=float),
        delta_C=df["delta_C"].to_numpy(dtype=float),
    )
    return series, t


# ── Figure 1: ΔΦ(t) with threshold lines ─────────────────────────────────────

def plot_delta_phi_thresholds(
    t: np.ndarray,
    delta_phi: np.ndarray,
    theta_low: float,
    theta_high: float,
    out_path: str,
) -> None:
    """Plot ΔΦ(t) with θ_low and θ_high threshold lines and save."""
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(t, delta_phi, color="#1565C0", linewidth=1.0, label=r"$\Delta\Phi(t)$")
    ax.axhline(
        theta_low,
        color="#FF9800",
        linewidth=1.2,
        linestyle="--",
        label=rf"$\theta_{{\mathrm{{low}}}}$ = {theta_low:.3f}",
    )
    ax.axhline(
        theta_high,
        color="#F44336",
        linewidth=1.2,
        linestyle="--",
        label=rf"$\theta_{{\mathrm{{high}}}}$ = {theta_high:.3f}",
    )

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel(r"$\Delta\Phi(t)$", fontsize=11)
    ax.set_title(
        r"Stage 5: Instability Magnitude $\Delta\Phi(t)$ with Regime Thresholds",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 2: Regime segmentation ────────────────────────────────────────────

def plot_regime_segmentation(
    t: np.ndarray,
    delta_phi: np.ndarray,
    regime_labels,
    out_path: str,
) -> None:
    """Plot broad regime bands with ΔΦ(t) overlaid and transition markers."""
    intervals = regime_shading_intervals(regime_labels)
    transitions = extract_transition_events(regime_labels)

    fig, ax = plt.subplots(figsize=(14, 5))

    # ── Broad colored regime bands ────────────────────────────────────────────
    dt_half = (t[1] - t[0]) / 2
    for iv in intervals:
        t_start = t[iv.start]
        t_end = t[iv.stop - 1] + dt_half if iv.stop < len(t) else t[-1] + dt_half
        ax.axvspan(
            t_start,
            t_end,
            alpha=REGIME_ALPHA,
            color=REGIME_COLORS.get(iv.regime, "#CCCCCC"),
            linewidth=0,
        )

    # ── Clean boundary lines at transitions ──────────────────────────────────
    for ev in transitions:
        ax.axvline(
            t[ev.index],
            color="#616161",
            linewidth=0.8,
            linestyle="--",
            alpha=0.6,
            zorder=3,
        )

    # ── ΔΦ(t) overlaid as a single curve ─────────────────────────────────────
    ax.plot(
        t, delta_phi,
        color="#212121",
        linewidth=1.0,
        zorder=4,
        label=r"$\Delta\Phi(t)$",
    )

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel(r"$\Delta\Phi(t)$", fontsize=11)
    ax.set_title(
        r"Stage 5: Regime Segmentation of $\Delta\Phi(t)$",
        fontsize=12,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    present_regimes = set(regime_labels)
    regime_patches = [
        Patch(
            facecolor=REGIME_COLORS[r],
            alpha=0.6,
            label=r.capitalize(),
        )
        for r in ["stable", "pre-instability", "instability", "recovery"]
        if r in present_regimes
    ]
    transition_handle = Line2D(
        [0], [0],
        color="#616161",
        linewidth=0.8,
        linestyle="--",
        alpha=0.6,
        label="Transition",
    )
    delta_phi_handle = Line2D(
        [0], [0],
        color="#212121",
        linewidth=1.0,
        label=r"$\Delta\Phi(t)$",
    )
    ax.legend(
        handles=[delta_phi_handle] + regime_patches + [transition_handle],
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(_FIG_DIR, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    series, t = _load_series(_DATA_PATH)

    # ── 2. Compute ΔΦ(t) ─────────────────────────────────────────────────────
    delta_phi = compute_delta_phi(series)

    # ── 3. Detect regimes with minimum dwell-time stability (10 samples) ──────
    regime_result = detect_regimes(delta_phi, min_dwell=10)

    # ── 4. Plot ΔΦ(t) with threshold lines ───────────────────────────────────
    plot_delta_phi_thresholds(
        t,
        delta_phi,
        theta_low=regime_result.theta_low,
        theta_high=regime_result.theta_high,
        out_path=os.path.join(_FIG_DIR, "delta_phi_thresholds.png"),
    )

    # ── 5. Plot regime segmentation ───────────────────────────────────────────
    plot_regime_segmentation(
        t,
        delta_phi,
        regime_labels=regime_result.labels,
        out_path=os.path.join(_FIG_DIR, "regime_segmentation.png"),
    )


if __name__ == "__main__":
    main()
