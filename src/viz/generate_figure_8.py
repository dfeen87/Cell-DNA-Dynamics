"""
Generate Figure 8 — Null-Test Comparison.

Produces a single publication-ready PNG comparing four ΔΦ(t) trajectories:
  1. Observed synthetic benchmark
  2. Shuffled-time null model
  3. Phase-randomized null model
  4. Stable control baseline

Output: figures/final/figure_8_null_comparison.png
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from src.simulation.generate_synthetic import generate_synthetic_ice  # noqa: E402
from src.simulation.null_tests import (  # noqa: E402
    shuffled_time_null,
    phase_randomized_null,
    stable_control_null,
)

# ---------------------------------------------------------------------------
# Manuscript color palette
# ---------------------------------------------------------------------------
_COLORS = {
    "observed":     "#C62828",   # deep crimson — observed signal
    "shuffled":     "#1565C0",   # deep blue — shuffled-time null
    "surrogate":    "#E65100",   # deep orange — phase-randomized null
    "stable":       "#2E7D32",   # deep green — stable control
}

# Observed curve is drawn thick and fully opaque to stand out clearly.
# Null curves are deliberately thinner and semi-transparent to reduce clutter.
_LW_OBSERVED = 2.8   # prominent
_LW_NULL     = 1.1   # unobtrusive

_ALPHA_SHUFFLED  = 0.45   # shuffled-time null – extra subdued (can have large swings)
_ALPHA_NULL      = 0.60   # other null curves


def generate_figure_8(seed: int = 0) -> str:
    """Generate and save Figure 8.

    Parameters
    ----------
    seed : int
        Random seed used for all surrogates (ensures reproducibility).

    Returns
    -------
    save_path : str
        Absolute path to the saved PNG file.
    """
    # ------------------------------------------------------------------
    # 1. Generate observed synthetic benchmark
    # ------------------------------------------------------------------
    series, t, _ = generate_synthetic_ice(seed=seed)
    delta_E = series.delta_E
    delta_I = series.delta_I
    delta_C = series.delta_C
    observed = np.sqrt(delta_E ** 2 + delta_I ** 2 + delta_C ** 2)

    # ------------------------------------------------------------------
    # 2. Generate null surrogates
    # ------------------------------------------------------------------
    shuffled  = shuffled_time_null(observed, seed=seed)
    surrogate = phase_randomized_null(delta_E, delta_I, delta_C, seed=seed)
    control   = stable_control_null(t, seed=seed)

    # ------------------------------------------------------------------
    # 3. Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))

    # Null curves drawn first (behind), thin and semi-transparent to reduce clutter.
    ax.plot(t, shuffled,  lw=_LW_NULL, color=_COLORS["shuffled"],
            label="Shuffled-time null",      zorder=2, alpha=_ALPHA_SHUFFLED)
    ax.plot(t, surrogate, lw=_LW_NULL, color=_COLORS["surrogate"],
            label="Phase-randomized null",   zorder=2, alpha=_ALPHA_NULL)
    ax.plot(t, control,   lw=_LW_NULL, color=_COLORS["stable"],
            label="Stable control",          zorder=2, alpha=_ALPHA_NULL)

    # Observed curve drawn last (on top), thick and fully opaque for prominence.
    ax.plot(t, observed,  lw=_LW_OBSERVED, color=_COLORS["observed"],
            label="Observed",                zorder=4)

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("ΔΦ(t)", fontsize=12)
    ax.set_title("Figure 8 — Null-Test Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.tick_params(labelsize=10)
    ax.set_xlim(t[0], t[-1])

    # Clip the y-axis to a range anchored on the observed signal so that
    # high-amplitude shuffled-time swings do not compress the rest of the plot.
    ax.set_ylim(bottom=0, top=np.max(observed) * 1.35)

    plt.tight_layout()

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    save_path = os.path.join(_REPO_ROOT, "figures", "final",
                             "figure_8_null_comparison.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    path = generate_figure_8()
    print(f"Saved → {path}")
