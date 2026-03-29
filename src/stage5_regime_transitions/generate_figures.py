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
    seed = 0, DPI = 300 (manuscript) / 150 (debug), figure sizes as specified per figure below.
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

# Use a clean sans-serif font throughout.
matplotlib.rcParams["font.family"] = "DejaVu Sans"

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
DPI       = 400   # manuscript / high-quality – 350-450 range for crisp print output
DPI_DEBUG = 150   # debug / quick-inspect output
np.random.seed(SEED)

# ── Consistent regime colour palette (Stage 2 / Stage 3 compatible) ──────────
# Colors are the original palette with ~12% saturation boost for vibrancy.
REGIME_COLORS = {
    "stable":          "#1597ff",   # blue  (+12% sat)
    "pre-instability": "#ff9800",   # orange (already max sat)
    "instability":     "#ff3a2b",   # red   (+12% sat)
    "recovery":        "#46b54b",   # green (+12% sat)
}
REGIME_ALPHA            = 0.18   # polished (manuscript) – avoids washout
REGIME_ALPHA_UNPOLISHED = 0.18   # debug – same value for reproducibility

# Minimum contiguous-block length (samples) for a transition marker to be shown
# in the polished figure.  Short-lived blocks within the same consolidated regime
# are suppressed.  Matches the min_dwell parameter used in detect_regimes.
_MIN_TRANSITION_BLOCK = 30

# ── Paths ─────────────────────────────────────────────────────────────────────
_DATA_PATH      = os.path.join(_REPO_ROOT, "data", "synthetic", "eic_timeseries.csv")
_FIG_DIR        = os.path.join(_REPO_ROOT, "figures", "stage5_regime_transitions")
_MANUSCRIPT_DIR = os.path.join(_FIG_DIR, "manuscript")
_DEBUG_DIR      = os.path.join(_FIG_DIR, "debug")


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
    *,
    polished: bool = True,
    extra_out_paths: list[str] | None = None,
) -> None:
    """Plot broad regime bands with ΔΦ(t) overlaid and transition markers.

    Parameters
    ----------
    polished : bool, default True
        When *True* (manuscript output) applies light visual refinements:
        softer regime-band alpha, thicker ΔΦ(t) curve, lighter/thinner
        transition lines, and suppression of short-lived transition markers.
        When *False* (debug output) the original visual parameters are used.
        Neither mode alters the underlying segmentation logic or thresholds.
    extra_out_paths : list of str, optional
        Additional output paths (e.g. PDF, SVG) to save alongside ``out_path``.
    """
    intervals   = regime_shading_intervals(regime_labels)
    transitions = extract_transition_events(regime_labels)

    # ── Visual parameters ─────────────────────────────────────────────────────
    if polished:
        regime_alpha    = REGIME_ALPHA           # 0.18 – avoids washout
        curve_lw        = 2.0                    # +0.2 thicker ΔΦ(t) for readability
        trans_color     = "#AAAAAA"              # light gray per spec
        trans_lw        = 0.6                    # thin line
        trans_alpha     = 0.9
        trans_dash      = (0, (4, 8))            # wider dash spacing for cleaner look
        curve_aa        = True                   # antialiasing enabled
        trans_aa        = True
        # Suppress transition markers for short-lived blocks (visual clutter).
        major_starts = {
            iv.start
            for iv in intervals
            if (iv.stop - iv.start) >= _MIN_TRANSITION_BLOCK and iv.start > 0
        }
        visible_transitions = [ev for ev in transitions if ev.index in major_starts]
    else:
        regime_alpha        = REGIME_ALPHA_UNPOLISHED   # 0.18 – original
        curve_lw            = 1.0
        trans_color         = "#616161"
        trans_lw            = 0.8
        trans_alpha         = 0.6
        trans_dash          = "--"
        curve_aa            = True
        trans_aa            = True
        visible_transitions = transitions

    fig, ax = plt.subplots(figsize=(14, 5))

    # ── Broad colored regime bands ────────────────────────────────────────────
    dt_half = (t[1] - t[0]) / 2
    for iv in intervals:
        t_start = t[iv.start]
        t_end = t[iv.stop - 1] + dt_half if iv.stop < len(t) else t[-1] + dt_half
        ax.axvspan(
            t_start,
            t_end,
            alpha=regime_alpha,
            color=REGIME_COLORS.get(iv.regime, "#CCCCCC"),
            linewidth=0,
        )

    # ── Boundary lines at (major) transitions ─────────────────────────────────
    for ev in visible_transitions:
        ax.axvline(
            t[ev.index],
            color=trans_color,
            linewidth=trans_lw,
            linestyle=trans_dash,
            alpha=trans_alpha,
            antialiased=trans_aa,
            zorder=3,
        )

    # ── ΔΦ(t) overlaid as a single curve ─────────────────────────────────────
    ax.plot(
        t, delta_phi,
        color="#000000",
        linewidth=curve_lw,
        antialiased=curve_aa,
        zorder=4,
        label=r"$\Delta\Phi(t)$",
    )

    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylabel(r"$\Delta\Phi(t)$", fontsize=13)
    ax.set_title(
        r"Stage 5: Regime Segmentation of $\Delta\Phi(t)$",
        fontsize=14,
    )
    ax.tick_params(axis="both", labelsize=11)

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
        color=trans_color,
        linewidth=trans_lw,
        linestyle=trans_dash,
        alpha=1.0,
        label="Transition",
    )
    delta_phi_handle = Line2D(
        [0], [0],
        color="#000000",
        linewidth=curve_lw,
        label=r"$\Delta\Phi(t)$",
    )
    ax.legend(
        handles=[delta_phi_handle] + regime_patches + [transition_handle],
        loc="upper right",
        fontsize=9,
        framealpha=1.0,
        edgecolor="#CCCCCC",
    )
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

    fig.tight_layout()
    save_dpi = DPI if polished else DPI_DEBUG
    fig.savefig(out_path, dpi=save_dpi, bbox_inches="tight")
    print(f"Saved: {out_path}")
    for extra_path in (extra_out_paths or []):
        fig.savefig(extra_path, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved: {extra_path}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(_FIG_DIR,        exist_ok=True)
    os.makedirs(_MANUSCRIPT_DIR, exist_ok=True)
    os.makedirs(_DEBUG_DIR,      exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    series, t = _load_series(_DATA_PATH)

    # ── 2. Compute ΔΦ(t) ─────────────────────────────────────────────────────
    delta_phi = compute_delta_phi(series)

    # ── 3. Detect regimes – hysteresis + 30-sample minimum dwell time ─────────
    regime_result = detect_regimes(delta_phi, min_dwell=30)

    # ── 4. Plot ΔΦ(t) with threshold lines ───────────────────────────────────
    plot_delta_phi_thresholds(
        t,
        delta_phi,
        theta_low=regime_result.theta_low,
        theta_high=regime_result.theta_high,
        out_path=os.path.join(_FIG_DIR, "delta_phi_thresholds.png"),
    )

    # ── 5. Plot regime segmentation – unpolished (debug, for reproducibility) ─
    plot_regime_segmentation(
        t,
        delta_phi,
        regime_labels=regime_result.labels,
        out_path=os.path.join(_DEBUG_DIR, "regime_segmentation_unpolished.png"),
        polished=False,
    )

    # ── 6. Plot regime segmentation – polished (manuscript) ───────────────────
    _ms_png = os.path.join(_MANUSCRIPT_DIR, "regime_segmentation.png")
    _ms_pdf = os.path.join(_MANUSCRIPT_DIR, "regime_segmentation.pdf")
    # Also write to the canonical figures/stage5_regime_transitions/ location.
    _out_png = os.path.join(_FIG_DIR, "regime_segmentation.png")
    _out_pdf = os.path.join(_FIG_DIR, "regime_segmentation.pdf")
    plot_regime_segmentation(
        t,
        delta_phi,
        regime_labels=regime_result.labels,
        out_path=_ms_png,
        polished=True,
        extra_out_paths=[_ms_pdf, _out_png, _out_pdf],
    )


if __name__ == "__main__":
    main()
